import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- CONFIG ---
original_images_dir = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_32"
labels_csv = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_32/song_artist_genre.csv"
output_base_dir = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_32_split"
splits = {"train": 0.8, "val": 0.1, "test": 0.1}  # 80/10/10 split


def create_dirs():
    """Create train/val/test directories if they don't exist"""
    for split in splits:
        split_dir = os.path.join(output_base_dir, split)
        os.makedirs(split_dir, exist_ok=True)


def copy_file(row, split):
    """Copy image file to its split folder"""
    src_path = os.path.join(original_images_dir, row['filename'])
    dest_dir = os.path.join(output_base_dir, split)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, row['filename'])

    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
    else:
        print(f"⚠️ Missing file: {row['filename']} (skipped)")


def safe_train_test_split(df, test_size, random_state, stratify_col):
    """Try stratified split, fallback to random if stratify fails"""
    try:
        return train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[stratify_col]
        )
    except ValueError as e:
        print(f"⚠️ Stratified split failed ({e}), falling back to random split.")
        return train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=None
        )


def split_dataset():
    """Split dataset into train, val, test and copy files"""
    df = pd.read_csv(labels_csv)
    print(f"✅ Loaded {len(df)} entries from CSV.")

    # Drop rows with missing data
    initial_len = len(df)
    df = df.dropna(subset=['genre', 'filename'])
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"⚠️ Dropped {dropped} rows with missing genre or filename.")

    # Count samples per genre
    class_counts = df['genre'].value_counts()
    rare_classes = class_counts[class_counts < 3].index.tolist()

    # Handle rare genres
    if rare_classes:
        print(f"⚠️ Found rare genres with <3 samples: {rare_classes}")
        df_rare = df[df['genre'].isin(rare_classes)]
        df_common = df[~df['genre'].isin(rare_classes)]
    else:
        df_rare = pd.DataFrame(columns=df.columns)
        df_common = df

    # Split common classes
    if len(df_common) > 0:
        df_train, df_temp = safe_train_test_split(
            df_common,
            test_size=(splits['val'] + splits['test']),
            random_state=42,
            stratify_col='genre'
        )
        test_size_ratio = splits['test'] / (splits['val'] + splits['test'])
        df_val, df_test = safe_train_test_split(
            df_temp,
            test_size=test_size_ratio,
            random_state=42,
            stratify_col='genre'
        )
    else:
        df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Add rare samples to train
    df_train = pd.concat([df_train, df_rare], ignore_index=True)

    datasets = {'train': df_train, 'val': df_val, 'test': df_test}

    for split, split_df in datasets.items():
        print(f"📂 {split.upper()} set: {len(split_df)} samples")
        split_csv_path = os.path.join(output_base_dir, f"{split}_labels.csv")
        split_df.to_csv(split_csv_path, index=False)
        print(f"💾 Saved {split} CSV to {split_csv_path}")

        # Copy image files
        for _, row in split_df.iterrows():
            copy_file(row, split)

    print("🎉 Dataset split complete!")


if __name__ == "__main__":
    create_dirs()
    split_dataset()
