-- pgvector extension + embedding column + index setup
-- 대상 테이블: products

-- 1) 확장 설치 (권한 필요)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) embedding 컬럼 추가 (이미 있으면 스킵)
ALTER TABLE products
  ADD COLUMN IF NOT EXISTS embedding vector(1536);

-- 3) 인덱스 생성 (둘 중 하나 선택)
-- 옵션 A: HNSW (실시간/일반적으로 추천, pgvector 0.5+에서 지원)
-- NOTE: 운영에서는 create index가 오래 걸릴 수 있으니 트래픽 낮을 때 수행 권장
CREATE INDEX IF NOT EXISTS idx_products_embedding_hnsw
  ON products
  USING hnsw (embedding vector_cosine_ops);

-- 옵션 B: IVFFLAT (대량 데이터에서 튜닝 여지, ANALYZE 필요)
-- CREATE INDEX IF NOT EXISTS idx_products_embedding_ivfflat
--   ON products
--   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- ANALYZE products;

