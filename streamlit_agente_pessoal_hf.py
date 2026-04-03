import json
import os
import sqlite3
from datetime import date, datetime
from typing import Any

import numpy as np
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import HashingVectorizer


# =========================
# Configuração
# =========================
DEFAULT_MODEL = "openai/gpt-oss-20b:groq"
DEFAULT_SYSTEM_PROMPT = """
Você é um agente pessoal em português do Brasil.

Suas prioridades:
1. Ajudar o usuário e seu desenvolvedor Lucas Messias a organizar tarefas, prioridades e próximos passos.
2. Responder com objetividade e clareza.
3. Usar o contexto salvo no banco para personalizar a resposta.
4. Quando houver tarefas atrasadas, destacar isso com clareza.
5. Sempre que possível, sugerir um plano de ação prático.

Regras:
- Não invente informações fora do contexto fornecido.
- Se faltar contexto, diga isso claramente.
- Responda em português.
- Seja útil e direto.
""".strip()

DB_PATH = "agente_pessoal.db"
EMBED_DIM = 512
MAX_RELEVANT_CONTEXTS = 6
MAX_CHAT_HISTORY = 8


# =========================
# Helpers de ambiente
# =========================
def get_secret(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value:
        return value

    try:
        secret_value = st.secrets.get(key)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    return default


@st.cache_resource
def get_client() -> OpenAI:
    hf_token = get_secret("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN não configurado. Defina o token em Secrets no Streamlit Community Cloud."
        )

    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )


# =========================
# Banco SQLite
# =========================
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT NOT NULL,
            responsavel TEXT,
            status TEXT,
            prazo TEXT,
            observacao TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            content TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_id TEXT,
            chunk_text TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()

    if count_rows("tasks") == 0:
        seed_default_data(conn)

    if count_rows("chat_messages") == 0:
        add_chat_message(
            "assistant",
            "Olá! Sou seu agente pessoal. Posso resumir tarefas, priorizar atividades, sugerir próximos passos e montar planos de ação.",
        )

    if get_notes() is None:
        save_notes("")

    conn.close()


def count_rows(table_name: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) AS total FROM {table_name}")
    total = cur.fetchone()["total"]
    conn.close()
    return int(total)


def seed_default_data(conn: sqlite3.Connection) -> None:
    tasks = [
        {
            "titulo": "Teste",
            "responsavel": "Lucas",
            "status": "Em Andamento",
            "prazo": "",
            "observacao": "Definir proposta técnica inicial.",
        },
    ]

    cur = conn.cursor()
    for task in tasks:
        cur.execute(
            """
            INSERT INTO tasks (titulo, responsavel, status, prazo, observacao)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                task["titulo"],
                task["responsavel"],
                task["status"],
                task["prazo"],
                task["observacao"],
            ),
        )
    conn.commit()


def get_tasks() -> list[dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, titulo, responsavel, status, prazo, observacao, created_at, updated_at
        FROM tasks
        ORDER BY id DESC
        """
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def add_task(titulo: str, responsavel: str, status: str, prazo: str, observacao: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tasks (titulo, responsavel, status, prazo, observacao, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        ,
        (titulo, responsavel, status, prazo, observacao),
    )
    task_id = cur.lastrowid
    conn.commit()
    conn.close()

    chunk = (
        f"Tarefa: {titulo} | Responsável: {responsavel} | Status: {status} | "
        f"Prazo: {prazo or 'Sem prazo'} | Observação: {observacao or 'Sem observação'}"
    )
    upsert_memory_chunk("task", str(task_id), chunk)


def update_task_status(task_id: int, new_status: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE tasks
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (new_status, task_id),
    )
    conn.commit()
    cur.execute(
        "SELECT titulo, responsavel, status, prazo, observacao FROM tasks WHERE id = ?",
        (task_id,),
    )
    row = cur.fetchone()
    conn.close()

    if row:
        chunk = (
            f"Tarefa: {row['titulo']} | Responsável: {row['responsavel']} | Status: {row['status']} | "
            f"Prazo: {row['prazo'] or 'Sem prazo'} | Observação: {row['observacao'] or 'Sem observação'}"
        )
        upsert_memory_chunk("task", str(task_id), chunk)


def delete_task(task_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    cur.execute("DELETE FROM memory_chunks WHERE source_type = 'task' AND source_id = ?", (str(task_id),))
    conn.commit()
    conn.close()


def save_notes(content: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO notes (id, content, updated_at)
        VALUES (1, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            content = excluded.content,
            updated_at = CURRENT_TIMESTAMP
        """,
        (content,),
    )
    conn.commit()
    conn.close()

    upsert_memory_chunk("notes", "1", f"Anotações do usuário: {content or 'Sem anotações adicionais.'}")


def get_notes() -> str | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT content FROM notes WHERE id = 1")
    row = cur.fetchone()
    conn.close()
    return None if row is None else (row["content"] or "")


def add_chat_message(role: str, content: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_messages (role, content) VALUES (?, ?)",
        (role, content),
    )
    message_id = cur.lastrowid
    conn.commit()
    conn.close()

    if role in {"user", "assistant"}:
        upsert_memory_chunk("chat", str(message_id), f"{role.upper()}: {content}")


def get_chat_messages(limit: int | None = None) -> list[dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT id, role, content, created_at FROM chat_messages ORDER BY id ASC"
    params: tuple[Any, ...] = ()

    if limit is not None:
        query = (
            "SELECT * FROM ("
            "SELECT id, role, content, created_at FROM chat_messages ORDER BY id DESC LIMIT ?"
            ") sub ORDER BY id ASC"
        )
        params = (limit,)

    cur.execute(query, params)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def clear_chat() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chat_messages")
    cur.execute("DELETE FROM memory_chunks WHERE source_type = 'chat'")
    conn.commit()
    conn.close()

    add_chat_message(
        "assistant",
        "Conversa limpa. Posso te ajudar com prioridades, tarefas e planos de ação.",
    )


# =========================
# Vetorização local
# =========================
@st.cache_resource
def get_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=EMBED_DIM,
        alternate_sign=False,
        norm=None,
        lowercase=True,
        ngram_range=(1, 2),
    )


def embed_text(text: str) -> np.ndarray:
    vectorizer = get_vectorizer()
    sparse = vectorizer.transform([text])
    dense = sparse.toarray()[0].astype(np.float32)
    norm = np.linalg.norm(dense)
    if norm > 0:
        dense = dense / norm
    return dense


def vector_to_json(vector: np.ndarray) -> str:
    return json.dumps(vector.tolist(), ensure_ascii=False)


def json_to_vector(value: str) -> np.ndarray:
    return np.array(json.loads(value), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def upsert_memory_chunk(source_type: str, source_id: str, chunk_text: str) -> None:
    embedding = embed_text(chunk_text)
    embedding_json = vector_to_json(embedding)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id FROM memory_chunks
        WHERE source_type = ? AND source_id = ?
        LIMIT 1
        """,
        (source_type, source_id),
    )
    existing = cur.fetchone()

    if existing:
        cur.execute(
            """
            UPDATE memory_chunks
            SET chunk_text = ?, embedding_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (chunk_text, embedding_json, existing["id"]),
        )
    else:
        cur.execute(
            """
            INSERT INTO memory_chunks (source_type, source_id, chunk_text, embedding_json)
            VALUES (?, ?, ?, ?)
            """,
            (source_type, source_id, chunk_text, embedding_json),
        )

    conn.commit()
    conn.close()


def rebuild_memory_index() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM memory_chunks")
    conn.commit()
    conn.close()

    for task in get_tasks():
        upsert_memory_chunk(
            "task",
            str(task["id"]),
            (
                f"Tarefa: {task['titulo']} | Responsável: {task['responsavel']} | Status: {task['status']} | "
                f"Prazo: {task.get('prazo') or 'Sem prazo'} | Observação: {task.get('observacao') or 'Sem observação'}"
            ),
        )

    notes = get_notes() or ""
    upsert_memory_chunk("notes", "1", f"Anotações do usuário: {notes or 'Sem anotações adicionais.'}")

    for msg in get_chat_messages():
        upsert_memory_chunk("chat", str(msg["id"]), f"{msg['role'].upper()}: {msg['content']}")


def search_relevant_contexts(query: str, limit: int = MAX_RELEVANT_CONTEXTS) -> list[dict[str, Any]]:
    query_vector = embed_text(query)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, source_type, source_id, chunk_text, embedding_json, created_at, updated_at
        FROM memory_chunks
        """
    )
    rows = cur.fetchall()
    conn.close()

    scored: list[dict[str, Any]] = []
    for row in rows:
        chunk_vector = json_to_vector(row["embedding_json"])
        score = cosine_similarity(query_vector, chunk_vector)
        scored.append(
            {
                "id": row["id"],
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                "chunk_text": row["chunk_text"],
                "score": score,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return [item for item in scored[:limit] if item["score"] > 0.05]


# =========================
# Regras de tarefas
# =========================
def normalize_date(value: str) -> date | None:
    if not value:
        return None

    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def task_deadline_status(task: dict[str, Any]) -> str:
    status = task.get("status", "")
    if status == "Finalizado":
        return "Finalizado"

    prazo = normalize_date(task.get("prazo", ""))
    if not prazo:
        return "Sem prazo"

    today = date.today()
    if prazo < today:
        return "Atrasada"
    if prazo == today:
        return "Vence hoje"
    return "No prazo"


# =========================
# Contexto para o modelo
# =========================
def build_context_text(user_message: str) -> str:
    tasks = get_tasks()
    notes = (get_notes() or "").strip()
    contexts = search_relevant_contexts(user_message)

    lines = []
    lines.append("CONTEXTO GERAL:")
    lines.append("Você está respondendo com base em tarefas, anotações e histórico de conversa salvos no banco SQLite.")
    lines.append("")

    lines.append("RESUMO DE TAREFAS:")
    if tasks:
        for idx, task in enumerate(tasks, start=1):
            lines.append(
                f"{idx}. Título: {task['titulo']} | Responsável: {task['responsavel']} | "
                f"Status: {task['status']} | Situação do prazo: {task_deadline_status(task)} | "
                f"Prazo: {task.get('prazo') or 'Sem prazo'} | "
                f"Observação: {task.get('observacao') or 'Sem observação'}"
            )
    else:
        lines.append("Nenhuma tarefa cadastrada.")

    lines.append("")
    lines.append("ANOTAÇÕES DO USUÁRIO:")
    lines.append(notes if notes else "Sem anotações adicionais.")

    lines.append("")
    lines.append("CONTEXTOS MAIS RELEVANTES RECUPERADOS POR SIMILARIDADE:")
    if contexts:
        for idx, item in enumerate(contexts, start=1):
            lines.append(
                f"{idx}. [score={item['score']:.3f}] {item['chunk_text']}"
            )
    else:
        lines.append("Nenhum contexto semanticamente relevante encontrado.")

    return "\n".join(lines)


# =========================
# Chamada ao modelo
# =========================
def ask_agent(user_message: str) -> str:
    client = get_client()
    model = get_secret("HF_MODEL", DEFAULT_MODEL)
    system_prompt = get_secret("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    context_text = build_context_text(user_message)
    recent_messages = get_chat_messages(limit=MAX_CHAT_HISTORY)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "developer",
            "content": f"Use este contexto interno para responder:\n\n{context_text}",
        },
    ]

    for msg in recent_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=900,
    )

    return response.choices[0].message.content.strip()


# =========================
# Interface
# =========================
def configure_page() -> None:
    st.set_page_config(
        page_title="Agente Pessoal",
        page_icon="🤖",
        layout="wide",
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 7rem;
                max-width: 1200px;
            }

            .chat-shell {
                min-height: 62vh;
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
            }

            .chat-history {
                min-height: 45vh;
            }

            div[data-testid="stChatInput"] {
                position: sticky;
                bottom: 0.75rem;
                background: rgba(255,255,255,0.96);
                padding-top: 0.5rem;
                z-index: 100;
            }

            @media (prefers-color-scheme: dark) {
                div[data-testid="stChatInput"] {
                    background: rgba(14,17,23,0.96);
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    st.sidebar.title("⚙️ Painel")
    st.sidebar.caption("Ajustes do agente e gestão rápida de tarefas.")

    with st.sidebar.expander("Nova tarefa", expanded=True):
        with st.form("new_task_form", clear_on_submit=True):
            titulo = st.text_input("Título")
            responsavel = st.text_input("Responsável", value="Lucas")
            status = st.selectbox(
                "Status",
                ["Pendente", "Em Andamento", "Aguardando", "Finalizado"],
                index=1,
            )
            prazo = st.text_input("Prazo", placeholder="YYYY-MM-DD ou DD/MM/YYYY")
            observacao = st.text_area("Observação")
            submitted = st.form_submit_button("Adicionar tarefa", use_container_width=True)

            if submitted:
                if not titulo.strip():
                    st.sidebar.error("Informe o título da tarefa.")
                else:
                    add_task(
                        titulo=titulo.strip(),
                        responsavel=responsavel.strip() or "Não definido",
                        status=status,
                        prazo=prazo.strip(),
                        observacao=observacao.strip(),
                    )
                    st.sidebar.success("Tarefa adicionada.")
                    st.rerun()

    with st.sidebar.expander("Anotações pessoais", expanded=True):
        current_notes = get_notes() or ""
        notes = st.text_area(
            "Notas que o agente deve considerar",
            value=current_notes,
            height=180,
            key="notes_area",
        )
        if st.button("Salvar anotações", use_container_width=True):
            save_notes(notes)
            st.sidebar.success("Anotações salvas.")
            st.rerun()

    with st.sidebar.expander("Exportar / importar contexto"):
        payload = {
            "tasks": get_tasks(),
            "notes": get_notes() or "",
            "chat_messages": get_chat_messages(),
        }
        st.download_button(
            "Baixar contexto JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="agente_pessoal_contexto.json",
            mime="application/json",
            use_container_width=True,
        )

        uploaded = st.file_uploader("Importar contexto JSON", type=["json"])
        if uploaded is not None:
            try:
                data = json.load(uploaded)

                conn = get_connection()
                cur = conn.cursor()
                cur.execute("DELETE FROM tasks")
                cur.execute("DELETE FROM chat_messages")
                cur.execute("DELETE FROM memory_chunks")
                conn.commit()
                conn.close()

                for task in data.get("tasks", []):
                    add_task(
                        titulo=task.get("titulo", "Sem título"),
                        responsavel=task.get("responsavel", "Não definido"),
                        status=task.get("status", "Pendente"),
                        prazo=task.get("prazo", ""),
                        observacao=task.get("observacao", ""),
                    )

                save_notes(data.get("notes", ""))

                imported_messages = data.get("chat_messages", [])
                if imported_messages:
                    for msg in imported_messages:
                        role = msg.get("role", "assistant")
                        content = msg.get("content", "")
                        if content:
                            add_chat_message(role, content)
                else:
                    add_chat_message(
                        "assistant",
                        "Contexto importado com sucesso. Posso continuar de onde você parou.",
                    )

                rebuild_memory_index()
                st.sidebar.success("Contexto importado com sucesso.")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Erro ao importar JSON: {exc}")

    if st.sidebar.button("Reindexar memória vetorial", use_container_width=True):
        rebuild_memory_index()
        st.sidebar.success("Memória vetorial reconstruída.")
        st.rerun()

    if st.sidebar.button("Limpar conversa", use_container_width=True):
        clear_chat()
        st.rerun()


def render_metrics() -> None:
    tasks = get_tasks()
    total = len(tasks)
    atrasadas = sum(1 for task in tasks if task_deadline_status(task) == "Atrasada")
    abertas = sum(1 for task in tasks if task.get("status") != "Finalizado")
    finalizadas = sum(1 for task in tasks if task.get("status") == "Finalizado")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Em aberto", abertas)
    c3.metric("Finalizadas", finalizadas)
    c4.metric("Atrasadas", atrasadas)


def render_tasks_tab() -> None:
    st.subheader("Tarefas")

    tasks = get_tasks()
    if not tasks:
        st.info("Nenhuma tarefa cadastrada.")
        return

    for task in tasks:
        situacao = task_deadline_status(task)
        with st.container(border=True):
            st.markdown(f"**{task['titulo']}**")
            st.write(
                f"Responsável: {task['responsavel']} | Status: {task['status']} | Situação: {situacao} | Prazo: {task.get('prazo') or 'Sem prazo'}"
            )
            if task.get("observacao"):
                st.caption(task["observacao"])

            col1, col2 = st.columns(2)
            with col1:
                if task["status"] != "Finalizado":
                    if st.button("Marcar como finalizada", key=f"done_{task['id']}", use_container_width=True):
                        update_task_status(task["id"], "Finalizado")
                        st.rerun()
            with col2:
                if st.button("Excluir", key=f"delete_{task['id']}", use_container_width=True):
                    delete_task(task["id"])
                    st.rerun()


def render_chat_tab() -> None:
    st.subheader("Chat com o agente")

    quick1, quick2, quick3 = st.columns(3)
    if quick1.button("Priorize minhas tarefas", use_container_width=True):
        st.session_state.quick_prompt = "Priorize minhas tarefas e diga o que eu deveria atacar primeiro hoje."
    if quick2.button("Resuma atrasos", use_container_width=True):
        st.session_state.quick_prompt = "Quais tarefas estão atrasadas e qual plano de ação você sugere?"
    if quick3.button("Montar plano semanal", use_container_width=True):
        st.session_state.quick_prompt = "Monte um plano semanal com base nas minhas tarefas atuais."

    messages = get_chat_messages()

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Pergunte algo ao seu agente pessoal")
    if not prompt and st.session_state.get("quick_prompt"):
        prompt = st.session_state.pop("quick_prompt")

    if prompt:
        add_chat_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    answer = ask_agent(prompt)
                except Exception as exc:
                    answer = (
                        "Não consegui consultar o modelo agora. "
                        f"Detalhe técnico: {exc}"
                    )
                st.markdown(answer)

        add_chat_message("assistant", answer)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_about_tab() -> None:
    st.subheader("Como configurar no Streamlit Community Cloud")
    st.markdown(
        """
        **Secrets obrigatórios**

        ```toml
        HF_TOKEN = "seu_token_hugging_face"
        HF_MODEL = "openai/gpt-oss-20b:groq"
        ```

        **Dependências sugeridas**

        ```txt
        streamlit
        openai
        numpy
        scikit-learn
        ```

        **O que mudou nesta versão**

        - Persistência em banco SQLite.
        - Chat salvo em disco.
        - Tarefas e anotações persistidas.
        - Memória vetorial local com HashingVectorizer + similaridade cosseno.
        - Reindexação manual da memória.
        - Ajuste visual para manter o input do chat no rodapé.
        """
    )


def main() -> None:
    configure_page()
    inject_css()
    init_db()

    st.title("🤖 Agente Pessoal")
    st.caption("Assistente pessoal com tarefas, anotações e chat usando LLM público, SQLite e memória vetorial local.")

    render_sidebar()
    render_metrics()
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Chat", "Tarefas", "Configuração"])
    with tab1:
        render_chat_tab()
    with tab2:
        render_tasks_tab()
    with tab3:
        render_about_tab()


if __name__ == "__main__":
    main()
