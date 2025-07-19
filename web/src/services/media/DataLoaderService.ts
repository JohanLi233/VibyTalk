/**
 * @file 图像数据加载服务
 *
 * @author JohanLi233
 */

import JSZip from "jszip";

export interface ImageMetadata {
  frame_id: string;
  full_image: string;
  face_image: string;
  tensor_file: string;
  crop_info: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
    width: number;
  };
}

export interface DatasetInfo {
  source_image_dimensions?: { width: number; height: number };
  config: {
    crop_size: number;
    mask_region: [number, number, number, number];
  };
}

export interface ImageDataResponse {
  images: ImageMetadata[];
  dataset_info: DatasetInfo;
}

/**
 * Load image metadata and image zip file
 * @returns object containing image metadata, JSZip instance, and zip Blob
 * @throws error if any resource loading fails
 */
export async function loadImageData(): Promise<{
  imageData: ImageDataResponse;
  zip: JSZip;
  zipBlob: Blob;
}> {
  const [imageDataResponse, imageZipResponse] = await Promise.all([
    fetch("/complete_dataset.json"),
    fetch("/processed_images.zip"),
  ]);

  if (!imageDataResponse.ok) {
    throw new Error("Failed to load image metadata `complete_dataset.json`");
  }
  if (!imageZipResponse.ok) {
    throw new Error("Failed to load image zip file `processed_images.zip`");
  }

  const imageData = (await imageDataResponse.json()) as ImageDataResponse;

  const zipBlob = await imageZipResponse.blob();
  const zip = await JSZip.loadAsync(zipBlob);

  return { imageData, zip, zipBlob };
}
