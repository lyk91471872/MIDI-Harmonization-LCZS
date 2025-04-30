import os
import tarfile
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_extract_task(split, idx, target_dir, base_url):
    """
    Download a single .tar.bz2 shard and extract only the desired files:
    - Keep directory structure under track folder
    - Include .mid/.midi and metadata.yaml
    - Skip any .wav files and stems_audio folder entirely
    """
    split_dir = os.path.join(target_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    archive_name = f"{idx}.tar.bz2"
    archive_path = os.path.join(split_dir, archive_name)
    url = f"{base_url}/{split}/{archive_name}"

    # Download if missing
    if not os.path.exists(archive_path):
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(archive_path, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True,
            desc=f"Downloading {split}/{archive_name}", leave=True
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Extract allowed files, preserving directory structure
    with tarfile.open(archive_path, 'r:bz2') as tar:
        members = []
        for m in tar.getmembers():
            nm = m.name
            # Keep metadata.yaml at top level
            if os.path.basename(nm) == 'metadata.yaml':
                members.append(m)
            # Keep mix.mid or mix.midi in track_dir
            elif nm.lower().endswith('.mid'):
                # skip any midi under stems_audio/
                if 'stems_audio/' in nm:
                    continue
                # allow mix.mid and any under stems_midi/
                members.append(m)
            # all others (e.g. .wav) skipped
        for m in tqdm(members,
                    desc=f"Extracting {split}/{archive_name}", leave=True):
            # compute output path
            out_path = os.path.join(split_dir, m.name)
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            # extract file content
            fobj = tar.extractfile(m)
            if fobj:
                with open(out_path, 'wb') as out_f:
                    out_f.write(fobj.read())

    # Remove archive to save space
    os.remove(archive_path)


def download_and_extract_midi_only(
    target_dir="cocochorales",
    splits={"train": 96, "valid": 12, "test": 12},
    base_url=(
        "https://storage.googleapis.com/"
        "magentadata/datasets/cocochorales/"
        "cocochorales_full_v1_zipped/main_dataset"
    ),
    max_workers=8,
):
    """
    Concurrently download & extract main_dataset shards (MIDI-only),
    track completed shards in a txt file, and print 'Done' at the end.
    """
    os.makedirs(target_dir, exist_ok=True)
    track_file = os.path.join(target_dir, 'downloaded.txt')
    done = set()
    if os.path.exists(track_file):
        with open(track_file) as tf:
            done = set(line.strip() for line in tf if line.strip())

    tasks = [(split, idx) for split, count in splits.items() for idx in range(1, count+1)]
    pending = [(s, i) for s, i in tasks if f"{s}/{i}" not in done]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_extract_task, s, i, target_dir, base_url): (s, i)
                   for s, i in pending}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Overall download+extract progress"):
            split, idx = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error on {split}/{idx}: {e}")
            else:
                with open(track_file, 'a') as tf:
                    tf.write(f"{split}/{idx}\n")

    print("Done")

if __name__ == '__main__':
    download_and_extract_midi_only()
