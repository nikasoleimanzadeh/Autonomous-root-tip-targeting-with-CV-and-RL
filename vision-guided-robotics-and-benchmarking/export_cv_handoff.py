"""Run on school server to export only Task 7 CSV + crop metadata + JPG views.

Notebook-produced root coordinates remain in original crop pixels. JPG preview
images are resized for portable GIF evidence and MUST NOT be used as input to
pixel->world calibration; use the ORIGINAL crop width/height from metadata.
"""
from pathlib import Path
import zipfile
import cv2

OUTPUT = Path('/home/y2a/task8_clean_pipeline/outputs')
DEST = Path('/home/y2a/task12_cv_handoff.zip')
MAX_PREVIEW_DIM = 900


def main() -> None:
    table = OUTPUT / '07_primary_roots' / 'primary_root_lengths.csv'
    metadata = OUTPUT / 'petri_crop_metadata.csv'
    for p in (table, metadata):
        if not p.exists():
            raise FileNotFoundError(p)
    photos = sorted((OUTPUT/'01_petri_crops').glob('*_petri.png'))
    if not photos:
        raise FileNotFoundError('No saved Petri crop images in 01_petri_crops')
    with zipfile.ZipFile(DEST, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(table, 'primary_root_lengths.csv')
        z.write(metadata, 'petri_crop_metadata.csv')
        for img_path in photos:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f'Could not load {img_path}')
            h,w = img.shape[:2]
            scale = min(1.0, MAX_PREVIEW_DIM/max(h,w))
            preview = cv2.resize(img, (max(1,round(w*scale)),max(1,round(h*scale))),
                                 interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError(f'Cannot encode {img_path}')
            z.writestr('previews/'+img_path.stem+'.jpg', encoded.tobytes())
    print('Exported:', DEST, 'with', len(photos), 'crop previews')


if __name__ == '__main__':
    main()
