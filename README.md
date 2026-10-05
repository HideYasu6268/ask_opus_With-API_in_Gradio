# 個人相談エージェント

## ローカル起動
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=... GEMINI_API_KEY=...        # Geminiは有料枠のキー
export GOOGLE_SERVICE_ACCOUNT_JSON="$(cat sa-key.json)" # ローカル用。DriveのファイルをこのSAのメールに共有しておく
python app.py   # http://localhost:8080
```

## Cloud Run + IAP（仕様書12・13・14章）
```bash
gcloud run deploy consult-agent --source . --region asia-northeast1 \
  --max-instances 1 --min-instances 0 --no-allow-unauthenticated --iap \
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest,GEMINI_API_KEY=gemini-key:latest
# 自分のGoogleアカウントにIAPアクセス権（roles/iap.httpsResourceAccessor）を付与
```
Drive認証: 実行SAのADCで `drive.readonly` が使えなければ、SAキーをSecret Managerに置き
`GOOGLE_SERVICE_ACCOUNT_JSON` として渡す。

## 環境変数
`CACHE_TTL`(1h/5m) / `GEMINI_MODEL` / `USD_JPY` / `COMPRESS_CHARS` / `KEEP_TURNS` / `APP_USER`・`APP_PASSWORD`（任意の二重認証）
単価・モデルIDは `config.py`。
