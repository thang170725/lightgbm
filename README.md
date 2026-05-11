# LightGBM Electricity Forecast - Huong Dan Chay Du An

README nay huong dan chay du an tu dau den cuoi:
- Chuan bi dataset
- Tien xu ly du lieu
- Train va luu model
- Chay backend API
- Chay frontend

## 1) Yeu cau moi truong

- Python 3.10+ (khuyen nghi 3.10/3.11)
- Node.js 18+ (de chay frontend Next.js)
- `pip`, `npm`

## 2) Cai dat dependencies

Tai thu muc goc du an:

```bash
pip install -r requirements.txt
cd frontend && npm install
```

## 3) Tai dataset goc

1. Tai dataset tai:
   [UCI Electricity Load Diagrams 2011-2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
2. Dat file `LD2011_2014.txt` vao:
   - `backend/dataset/LD2011_2014.txt`

Sau buoc nay, duong dan file phai ton tai dung nhu sau:

```bash
backend/dataset/LD2011_2014.txt
```

## 4) Tien xu ly du lieu de train

### Buoc 4.1 - Tao hourly dataset

Chay:

```bash
python -m backend.app.utils.build_hourly_dataset
```

Ket qua mong doi:
- Tao file `backend/dataset/hourly_electricity_filtered.csv`

### Buoc 4.2 - Tao train/valid/test

Chay:

```bash
python -m backend.app.utils.preprocessor
```

Ket qua mong doi:
- `backend/dataset/train_ready.csv`
- `backend/dataset/valid_ready.csv`
- `backend/dataset/test_ready.csv`

## 5) Train model va luu model

Chay:

```bash
python -m backend.test
```

Ket qua mong doi:
- Model duoc train xong khong loi
- Tao file `backend/models/lightgbm_model.pkl`

## 6) Tao du lieu user mau (de test predict)

Chay:

```bash
python -m backend.app.utils.gen_data_user
```

Ket qua mong doi:
- Tao file `backend/dataset/user_data.csv`

## 7) Chay pipeline predict bang script

Chay:

```bash
python -m backend.main
```

Script se:
- Load model da luu
- Doc `backend/dataset/user_data.csv`
- Du doan 24 gio tiep theo
- In ket qua timeline ra terminal

## 8) Chay backend API

Chay o thu muc goc du an:

```bash
python -m uvicorn backend.application:app --reload
```

Backend API mac dinh:
- `http://127.0.0.1:8000`
- Endpoint predict: `POST /predict` (upload file CSV gom 2 cot: `time`, `value`)

## 9) Chay frontend

Mo terminal moi:

```bash
cd frontend
npm run dev
```

Frontend mac dinh:
- `http://localhost:3000`

## 10) Thu tu chay nhanh (tom tat)

Neu ban chay tu dau:

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
python -m backend.app.utils.build_hourly_dataset
python -m backend.app.utils.preprocessor
python -m backend.test
python -m backend.app.utils.gen_data_user
python -m backend.main
python -m uvicorn backend.application:app --reload
```

Sau do mo terminal khac:

```bash
cd frontend && npm run dev
```

## 11) Loi thuong gap

- `FileNotFoundError: backend/dataset/LD2011_2014.txt`
  - Kiem tra da dat dung ten file `LD2011_2014.txt` trong `backend/dataset/`.

- `Model not found: backend/models/lightgbm_model.pkl`
  - Ban chua chay `python -m backend.test` de train va luu model.

- API bao can it nhat 168 rows
  - File user upload phai co it nhat 168 dong du lieu lich su.

- Frontend khong goi duoc backend
  - Kiem tra backend dang chay (`uvicorn`) va dung port `8000`.