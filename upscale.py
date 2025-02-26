from PIL import Image

def upscale_image_no_interpolation(input_path, output_path, scale_factor):
    # 讀取圖片
    img = Image.open(input_path)
    
    # 計算新的尺寸
    new_size = (img.width * scale_factor, img.height * scale_factor)
    
    # 使用最近鄰插值來確保像素保持原樣
    img_upscaled = img.resize(new_size, Image.NEAREST)
    
    # 儲存結果
    img_upscaled.save(output_path)
    print(f"圖片已放大 {scale_factor} 倍，儲存為 {output_path}")

# 測試函數
upscale_image_no_interpolation("/home/shijie/Documents/Dreaminit/DreamInit/workspace/test/unet_process/1900/total_attn_map/9--o-.png", "output.png", 15)  # 放大 3 倍