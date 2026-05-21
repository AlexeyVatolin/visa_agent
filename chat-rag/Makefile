CHROMA_DB_URL := 'https://storage.yandexcloud.net/ai-visa/chroma_db.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=YCAJE4o7Ul_CBprgFUlN9JEfB%2F20260521%2Fru-central1%2Fs3%2Faws4_request&X-Amz-Date=20260521T130738Z&X-Amz-Expires=2592000&X-Amz-Signature=b3409ceb0ef54125a610ea4e606a46452ef33ec5b4e904d4cfb03db27b246ee7&X-Amz-SignedHeaders=host&response-content-disposition=attachment'

.PHONY: chroma-db

chroma-db:
	curl -sL $(CHROMA_DB_URL) -o chroma_db.zip
	rm -rf chroma_db __MACOSX
	unzip -o chroma_db.zip
	rm -rf chroma_db.zip __MACOSX
