from apscheduler.schedulers.blocking import BlockingScheduler
import logging
from dao.dao import update_dataset
from dotenv import load_dotenv
import  os
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
def job():
    try:
        logger.info("Tiến hành chạy job cập nhật dataset")
        update_dataset(os.getenv("URL_UPLOAD"))
        logger.info("Đã tiến thành update dataset thành công.")
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy job: {e}")

def start_scheduler():
    logger.info("Chạy lần đầu tiên ngay khi khởi động...")
    job()

    scheduler = BlockingScheduler()
    scheduler.add_job(job, 'cron', hour="*", minute="0")
    logger.info("Bắt đầu APScheduler. Job sẽ lặp lại mỗi 1 giờ.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Dừng scheduler.")





