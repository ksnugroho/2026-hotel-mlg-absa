"""
Export dataset dan anotasi dari Label Studio project ID 2.
Menggunakan bulk export API (satu request) lalu majority vote untuk
resolusi konflik antar-annotator.

Output:
  data/raw/dataset.csv       - semua task + label hasil majority vote
  data/raw/annotations.csv   - semua anotasi mentah (per annotator)
"""

import requests
import pandas as pd
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ['LABEL_STUDIO_URL']
REFRESH_TOKEN = os.environ['LABEL_STUDIO_REFRESH_TOKEN']
PROJECT_ID = int(os.environ['LABEL_STUDIO_PROJECT_ID'])
ASPECTS = ['kebersihan', 'pelayanan', 'fasilitas', 'lokasi', 'harga', 'kenyamanan_kamar', 'overall_sentiment']


def get_access_token():
    r = requests.post(f'{BASE_URL}/api/token/refresh/', json={'refresh': REFRESH_TOKEN})
    r.raise_for_status()
    return r.json()['access']


def bulk_export(headers):
    print('Mengunduh semua data via export API (satu request)...')
    r = requests.get(
        f'{BASE_URL}/api/projects/{PROJECT_ID}/export',
        params={'exportType': 'JSON'},
        headers=headers,
        timeout=300
    )
    r.raise_for_status()
    data = r.json()
    print(f'  Selesai: {len(data)} task diterima.')
    return data


def parse_annotation(annotation):
    completed_by = annotation.get('completed_by')
    if isinstance(completed_by, dict):
        annotator = completed_by.get('email')
    else:
        annotator = completed_by  # integer user_id pada bulk export
    result = {
        'annotator': annotator,
        'annotation_id': annotation['id'],
    }
    for item in annotation.get('result', []):
        aspect = item.get('from_name')
        choices = item['value'].get('choices', [])
        if aspect in ASPECTS and choices:
            result[aspect] = choices[0]
    return result


def majority_vote(labels):
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def main():
    os.makedirs('data/raw', exist_ok=True)

    token = get_access_token()
    headers = {'Authorization': f'Bearer {token}'}

    tasks = bulk_export(headers)

    dataset_rows = []
    annotation_rows = []

    print(f'Memproses {len(tasks)} task...')
    for task in tasks:
        task_id = task['id']
        data = task['data']
        annotations = task.get('annotations', [])

        aspect_votes = {asp: [] for asp in ASPECTS}
        for ann in annotations:
            if ann.get('was_cancelled'):
                continue
            parsed = parse_annotation(ann)
            for asp in ASPECTS:
                if asp in parsed:
                    aspect_votes[asp].append(parsed[asp])

            annotation_rows.append({
                'task_id': task_id,
                'Kecamatan': data.get('Kecamatan'),
                'Nama Hotel': data.get('Nama Hotel'),
                'Review': data.get('Review'),
                **{asp: parsed.get(asp) for asp in ASPECTS},
                'annotator': parsed.get('annotator'),
                'annotation_id': parsed.get('annotation_id'),
            })

        final_labels = {asp: majority_vote(aspect_votes[asp]) for asp in ASPECTS}
        dataset_rows.append({
            'task_id': task_id,
            'Kecamatan': data.get('Kecamatan'),
            'Nama Hotel': data.get('Nama Hotel'),
            'Review': data.get('Review'),
            'n_annotators': len([a for a in annotations if not a.get('was_cancelled')]),
            **final_labels,
        })

    df_dataset = pd.DataFrame(dataset_rows)
    df_annotations = pd.DataFrame(annotation_rows)

    df_dataset.to_csv('data/raw/dataset.csv', index=False)
    df_annotations.to_csv('data/raw/annotations.csv', index=False)

    print(f'\nSelesai!')
    print(f'  dataset.csv      : {len(df_dataset)} baris')
    print(f'  annotations.csv  : {len(df_annotations)} baris')
    print(f'\nDistribusi overall_sentiment:')
    print(df_dataset['overall_sentiment'].value_counts())


if __name__ == '__main__':
    main()
