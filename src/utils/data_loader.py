"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.
    Tự động cache vào thư mục data/faiss_cache để tái sử dụng, tiết kiệm quota API.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    from langchain_community.vectorstores import FAISS

    # Xác định cache directory theo model embedding để tránh lệch vector dimension
    model_name = getattr(embeddings, "model", getattr(embeddings, "model_name", "default"))
    safe_name = str(model_name).replace("/", "_").replace(":", "_")
    cache_path = Path(__file__).parent.parent.parent / "data" / f"faiss_cache_{safe_name}"

    if (cache_path / "index.faiss").exists():
        try:
            print(f"📦 Đang tải FAISS vectorstore từ cache ({cache_path.name}) ...")
            return FAISS.load_local(str(cache_path), embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Không thể tải cache ({e}), đang tạo lại FAISS index...")

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_path))
        print(f"💾 Đã lưu cache FAISS vào {cache_path.name}")
    except Exception as e:
        print(f"⚠️ Không thể lưu cache FAISS: {e}")

    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
