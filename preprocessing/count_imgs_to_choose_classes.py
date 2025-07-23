import os
from collections import defaultdict

def count_images_by_prefix(folder_path):
    # Define the range of prefixes
    prefix_range = [f"IP{str(i).zfill(3)}" for i in range(103)]

    # Dictionary to store counts for each prefix
    prefix_counts = defaultdict(int)

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file starts with the fixed "IP" prefix and matches the range
        if filename.startswith("IP") and len(filename) >= 5:  # Ensure it has at least "IPXXX" + more digits
            prefix_number = filename[2:5]
            if prefix_number.isdigit() and prefix_number in [f"{str(i).zfill(3)}" for i in range(103)]:
                prefix = f"IP{prefix_number}"
                prefix_counts[prefix] += 1

    return prefix_counts

def categorize_counts(counts):
    categories = {
        "-100": {"prefixes": [], "count": 0},
        "100-200": {"prefixes": [], "count": 0},
        "200-400": {"prefixes": [], "count": 0},
        "400-500": {"prefixes": [], "count": 0},
        "500-1000": {"prefixes": [], "count": 0},
        "+1000": {"prefixes": [], "count": 0}
    }

    for prefix, count in counts.items():
        if count < 100:
            categories["-100"]["count"] += 1
            categories["-100"]["prefixes"].append((prefix, count))
        elif 100 <= count < 200:
            categories["100-200"]["count"] += 1
            categories["100-200"]["prefixes"].append((prefix, count))
        elif 200 <= count < 400:
            categories["200-400"]["count"] += 1
            categories["200-400"]["prefixes"].append((prefix, count))
        elif 400 <= count < 500:
            categories["400-500"]["count"] += 1
            categories["400-500"]["prefixes"].append((prefix, count))
        elif 500 <= count <= 1000:  # Include counts exactly equal to 1000
            categories["500-1000"]["count"] += 1
            categories["500-1000"]["prefixes"].append((prefix, count))
        elif count > 1000:  # Strictly greater than 1000
            categories["+1000"]["count"] += 1
            categories["+1000"]["prefixes"].append((prefix, count))

    return categories


# Path to the folder
folder_path = r'F:\code_pfe_all\IP102_DATASET\annotated_images\JPEGImages\JPEGImages'  # Replace with your folder path

# Call the function and display the results
if __name__ == "__main__":
    counts = count_images_by_prefix(folder_path)
    categories = categorize_counts(counts)

    print("File counts by prefix:")
    for prefix, count in counts.items():
        print(f"The class {prefix}: {count} files")

    print("\nCategories:")
    for category, data in categories.items():
        print(f"{category}: {data['count']} prefixes")
        if data['prefixes']:
            print("  Details:")
            for prefix, count in data['prefixes']:
                print(f"    {prefix}: {count} files")
