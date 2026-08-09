import os
import shutil
import ssl
from icrawler.builtin import BingImageCrawler

# Disable SSL restrictions globally
ssl._create_default_https_context = ssl._create_unverified_context

TARGET_DIR = os.path.join("dataset_binary", "non_soil")
os.makedirs(TARGET_DIR, exist_ok=True)

# Count current files
existing_files = [
    f
    for f in os.listdir(TARGET_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]
current_count = len(existing_files)
target_total = 1015
needed_count = target_total - current_count

print(f"Current Non-Soil Images: {current_count}")
print(f"Target Non-Soil Images: {target_total}")

if needed_count <= 0:
    print("You already have enough non-soil images!")
else:
    print(f"Downloading {needed_count} new images using icrawler...\n")

    # Categories to scrape 
    queries = {
        "cars on road": 135,
        "modern city architecture": 135,
        "dogs pets": 130,
        "people standing group": 130,
    }

    temp_dir = "temp_icrawler_downloads"

    for query, limit in queries.items():
        print(f"\n---> Scraping category: '{query}' (Target: {limit} images)")
        save_folder = os.path.join(temp_dir, query.replace(" ", "_"))

        # Run crawler
        crawler = BingImageCrawler(
            storage={"root_dir": save_folder},
            downloader_threads=4,
            log_level=30,  # Hide debug logs
        )
        crawler.crawl(keyword=query, max_num=limit)

        # Move crawled images to dataset_binary/non_soil
        if os.path.exists(save_folder):
            for img_name in os.listdir(save_folder):
                src_path = os.path.join(save_folder, img_name)
                clean_name = f"icrawl_{query.replace(' ', '_')}_{img_name}"
                dst_path = os.path.join(TARGET_DIR, clean_name)

                try:
                    shutil.move(src_path, dst_path)
                except Exception:
                    pass

    # Cleanup temporary folder
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    final_files = [
        f
        for f in os.listdir(TARGET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    print("\n-------------------------------------------")
    print("Scraping Completed Successfully!")
    print(f"New Total Non-Soil Images Count: {len(final_files)}")
    print("-------------------------------------------")