from bing_image_downloader import downloader

queries = [
    "1972 Chevelle rear view",
    "1972 Chevelle back",
    "1972 Chevelle rear quarter",
    "1972 Chevelle SS from behind",
    "1972 Chevrolet Chevelle rear end",
]

for query in queries:
    print(f"\nSearching: {query}")
    downloader.download(
        query,
        limit=30,
        output_dir="dataset/raw/chevelle_rear",
        adult_filter_off=False,
        force_replace=False,
        timeout=60,
        filter="photo",
        verbose=True,
    )