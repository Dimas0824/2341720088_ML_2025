# ML_INFERENCE_API_FLASK

Aplikasi Flask untuk inference model ML, siap untuk deployment di Hugging Face Spaces.

## Cara Menjalankan Lokal

```markdown
pip install -r requirements.txt
python app.py
```

## Deployment Hugging Face Spaces

- Pastikan file model sudah di-upload.
- Spaces akan otomatis menjalankan `app.py` pada port 7860.

## Struktur Folder

- `app.py` : Entry point aplikasi Flask
- `templates/` : HTML templates (home, result)
- `static/` : File statis (CSS, JS)
- `requirements.txt` : Daftar dependency
