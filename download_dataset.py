from bing_image_downloader import downloader
import os

output_dir = "dataset/raw"

# Search queries - multiple variations to get diverse images
chevelle_queries = [
    "1972 Chevrolet Chevelle",
    "1972 Chevelle SS",
    "1972 Chevelle Malibu",
    "72 Chevelle coupe",
]

chevette_queries = [
    "1983 Chevrolet Chevette",
    "1983 Chevette hatchback",
    "Chevy Chevette 1983",
    "1983 Chevette sedan",
]

print("=== Downloading 1972 Chevelle images ===")
for query in chevelle_queries:
    print(f"\nSearching: {query}")
    downloader.download(
        query,
        limit=50,
        output_dir=os.path.join(output_dir, "chevelle_1972"),
        adult_filter_off=False,
        force_replace=False,
        timeout=60,
        filter="photo",
        verbose=True,
    )

print("\n=== Downloading 1983 Chevette images ===")
for query in chevette_queries:
    print(f"\nSearching: {query}")
    downloader.download(
        query,
        limit=50,
        output_dir=os.path.join(output_dir, "chevette_1983"),
        adult_filter_off=False,
        force_replace=False,
        timeout=60,
        filter="photo",
        verbose=True,
    )

# Count what we got
for car in ["chevelle_1972", "chevette_1983"]:
    car_dir = os.path.join(output_dir, car)
    total = 0
    if os.path.exists(car_dir):
        for folder in os.listdir(car_dir):
            folder_path = os.path.join(car_dir, folder)
            if os.path.isdir(folder_path):
                count = len([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                total += count
    print(f"\n{car}: {total} images downloaded")