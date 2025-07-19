import torch
import argparse
import numpy as np
import onnxruntime
import os
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from unet import UNet
from config import get_config, list_available_models, get_border_from_crop_size


def main():
    argument_parser = argparse.ArgumentParser(
        description="Convert PyTorch model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to PyTorch checkpoint file (.pth)",
    )
    argument_parser.add_argument(
        "--output", type=str, default="model.onnx", help="Path to save ONNX file"
    )
    argument_parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=list_available_models(),
        help="Model architecture size to use.",
    )
    argument_parser.add_argument(
        "--opset",
        type=int,
        default=20,
    )

    argument_parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization (INT8) to exported ONNX model",
    )

    parsed_args = argument_parser.parse_args()

    network_config = get_config(parsed_args.model_size)
    print(f"Using network configuration: {parsed_args.model_size}")

    input_resolution = network_config.input_resolution
    border_size = get_border_from_crop_size(input_resolution)
    effective_size = input_resolution - 2 * border_size

    batch_size = 1
    dummy_image_tensor = torch.randn(batch_size, 6, effective_size, effective_size, requires_grad=False)
    dummy_audio_tensor = torch.randn(
        batch_size,
        network_config.temporal_window_length,
        network_config.acoustic_group_count,
        network_config.acoustic_vector_size,
        requires_grad=False,
    )
    print(f"Dummy image input shape: {dummy_image_tensor.shape}")
    print(f"Dummy audio input shape: {dummy_audio_tensor.shape}")

    neural_network = UNet(n_channels=6, model_size=parsed_args.model_size)

    print(f"Loading checkpoint from: {parsed_args.checkpoint}")
    neural_network.load_state_dict(torch.load(parsed_args.checkpoint, map_location="cpu"))
    neural_network.eval()
    print("Model loaded successfully.")

    if parsed_args.quantize:
        quantized_output_path = parsed_args.output
        intermediate_name = quantized_output_path.rsplit(".", 1)[0]
        fp32_intermediate_path = f"{intermediate_name}.fp32.tmp.onnx"
        preprocessed_intermediate_path = f"{intermediate_name}.preprocessed.tmp.onnx"
        print(f"Will export quantized model (INT8): {quantized_output_path}")
    else:
        fp32_intermediate_path = parsed_args.output
        quantized_output_path = None
        preprocessed_intermediate_path = None

    try:
        total_steps = 4 if parsed_args.quantize else 2
        print(
            f"\nStep 1/{total_steps}: Exporting to FP32 ONNX model -> {fp32_intermediate_path}"
        )
        torch.onnx.export(
            neural_network,
            (dummy_image_tensor, dummy_audio_tensor),
            fp32_intermediate_path,
            input_names=["img_in", "audio_feat"],
            output_names=["output"],
            opset_version=parsed_args.opset,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            dynamic_axes={
                "img_in": {0: "batch_size"},
                "audio_feat": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        print("ONNX FP32 model export successful.")

        validation_target_path = fp32_intermediate_path
        if parsed_args.quantize:
            assert quantized_output_path is not None
            assert preprocessed_intermediate_path is not None

            print(
                f"\nStep 2/{total_steps}: Preprocessing model for quantization -> {preprocessed_intermediate_path}"
            )
            quant_pre_process(
                input_model_path=fp32_intermediate_path,
                output_model_path=preprocessed_intermediate_path,
            )
            print("✅ Model preprocessing completed.")

            print(
                f"\nStep 3/{total_steps}: Applying dynamic quantization -> {quantized_output_path}"
            )
            quantize_dynamic(
                model_input=preprocessed_intermediate_path,
                model_output=quantized_output_path,
                weight_type=QuantType.QUInt8,
            )
            validation_target_path = quantized_output_path
            print("✅ Dynamic quantization completed.")

        print(
            f"\nStep {total_steps}/{total_steps}: Validating model using onnxruntime: {validation_target_path}..."
        )

        with torch.no_grad():
            pytorch_output = neural_network(dummy_image_tensor, dummy_audio_tensor)

        onnx_session = onnxruntime.InferenceSession(
            validation_target_path, providers=["CPUExecutionProvider"]
        )

        onnx_inputs = {
            onnx_session.get_inputs()[0].name: dummy_image_tensor.numpy(),
            onnx_session.get_inputs()[1].name: dummy_audio_tensor.numpy(),
        }

        onnx_outputs = onnx_session.run(None, onnx_inputs)
        onnx_result = onnx_outputs[0]

        try:
            pytorch_result = pytorch_output.detach().cpu().numpy()
            if parsed_args.quantize:
                relative_tolerance, absolute_tolerance = 1e-1, 1e-1
                print(f"Using quantized model validation tolerance: rtol={relative_tolerance}, atol={absolute_tolerance}")
            else:
                relative_tolerance, absolute_tolerance = 1e-3, 1e-3

            np.testing.assert_allclose(
                pytorch_result, np.array(onnx_result), rtol=relative_tolerance, atol=absolute_tolerance
            )
            print("✅ Validation successful: PyTorch and ONNX model outputs match.")
        except AssertionError as validation_error:
            print("❌ Validation failed: PyTorch and ONNX model outputs do not match.")
            print(validation_error)
        finally:
            if parsed_args.quantize:
                if os.path.exists(fp32_intermediate_path):
                    os.remove(fp32_intermediate_path)
                    print(f"Removed temporary FP32 model: {fp32_intermediate_path}")
                if preprocessed_intermediate_path and os.path.exists(
                    preprocessed_intermediate_path
                ):
                    os.remove(preprocessed_intermediate_path)
                    print(f"Removed temporary preprocessed model: {preprocessed_intermediate_path}")

    except Exception as export_error:
        print(f"Error during ONNX export or validation: {export_error}")
        if parsed_args.quantize:
            if "fp32_intermediate_path" in locals() and os.path.exists(fp32_intermediate_path):
                os.remove(fp32_intermediate_path)
                print(f"Removed temporary FP32 model during error handling: {fp32_intermediate_path}")
            if (
                "preprocessed_intermediate_path" in locals()
                and preprocessed_intermediate_path
                and os.path.exists(preprocessed_intermediate_path)
            ):
                os.remove(preprocessed_intermediate_path)
                print(
                    f"Removed temporary preprocessed model during error handling: {preprocessed_intermediate_path}"
                )


if __name__ == "__main__":
    main()