import logging
import os
from pdf2zh.config import get_settings
from pdf2zh.e2e import get_parser

# Cấu hình log hiển thị ra terminal để xem quá trình load model
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_loading_configuration():
    print("=" * 60)
    print(" STAGE 1: KIỂM TRA THÔNG TIN TỪ BIẾN MÔI TRƯỜNG / .ENV")
    print("=" * 60)
    
    # 1. Lấy thông số đang được load thông qua Pydantic Settings
    settings = get_settings()
    
    print(f"📌 Trạng thái file .env: {'Có tồn tại' if os.path.exists('.env') else 'Không tồn tại (Đang dùng mặc định hệ thống)'}")
    print(f"🔹 DEVICE: {settings.device}")
    print(f"🔹 PAGE_BATCH_SIZE: {settings.page_batch_size}")
    print(f"🔹 LAYOUT_BATCH_SIZE: {settings.layout_batch_size}")
    print(f"🔹 DETECTION_BATCH_SIZE: {settings.detection_batch_size}")
    print(f"🔹 OCR_BATCH_SIZE: {settings.ocr_batch_size}")
    print(f"🔹 TABLE_BATCH_SIZE: {settings.table_batch_size}")
    print(f"🔹 DETECTOR_TEXT_THRESHOLD: {settings.detector_text_threshold}")
    print(f"🔹 DETECTOR_BLANK_THRESHOLD: {settings.detector_blank_threshold}")
    print("-" * 60)

    print("\n" + "=" * 60)
    print(" STAGE 2: KIỂM TRA KHỞI TẠO SINGLETON PARSER")
    print("=" * 60)
    
    # 2. Gọi get_parser để xem hệ thống có map các config này vào phần cứng không
    logger.info("Đang gọi get_parser()...")
    parser = get_parser()
    
    print("-" * 60)
    print("✅ Kiểm tra thuộc tính bên trong StageAParser sau khi map:")
    
    # Dump các giá trị phần cứng thực tế mà StageAParser đang nắm giữ sau khi qua hàm configure_settings
    if hasattr(parser, 'hardware'):
        print(f"⚙️ Cấu hình phần cứng thực tế (parser.hardware): {parser.hardware}")
    
    # Kiểm tra xem các threshold đã được đẩy vào OCR model chưa
    if hasattr(parser, 'ocr_model'):
        ocr = parser.ocr_model
        print(f"🔍 OCR Model Name: {getattr(ocr, 'model_name', 'Unknown')}")
        print(f"🎯 Ngưỡng Text thực tế trong instance: {getattr(ocr, 'detector_text_threshold', 'N/A')}")
        print(f"🎯 Ngưỡng Blank thực tế trong instance: {getattr(ocr, 'detector_blank_threshold', 'N/A')}")
    print("=" * 60)

if __name__ == "__main__":
    test_loading_configuration()