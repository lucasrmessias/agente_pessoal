import json
import os
from datetime import date, datetime
from typing import Any

import streamlit as st
from openai import OpenAI


# =========================
# Configuração
# =========================
DEFAULT_MODEL = "openai/gpt-oss-20b:groq"
DEFAULT_SYSTEM_PROMPT = """
Você é um agente pessoal em português do Brasil.

Suas prioridades:
1. Ajudar o usuário a organizar tarefas, prioridades e próximos passos.
2. Responder com objetividade e clareza.
3. Usar o contexto das tarefas e anotações para personalizar a resposta.
4. Quando houver tarefas atrasadas, destacar isso com clareza.
5. Sempre que possível, sugerir um plano de ação prático.

Regras:
- Não invente informações fora do contexto fornecido.
- Se faltar contexto, diga isso claramente.
- Responda em português.
- Seja útil e direto.
""".strip()


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
# Estado da aplicação
# =========================
def init_session_state() -> None:
    if "tasks" not in st.session_state:
        st.session_state.tasks = [
            {
                "titulo": "Validar Assistente Farma com transcrição de áudio",
                "responsavel": "Lucas",
                "status": "Em Andamento",
                "prazo": "",
                "observacao": "Validar fluxo ponta a ponta do áudio.",
            },
            {
                "titulo": "Separar monitoramento dados/integração",
                "responsavel": "Lucas",
                "status": "Em Andamento",
                "prazo": "",
                "observacao": "Criar visão separada para o monitoramento.",
            },
            {
                "titulo": "POC Centralização de Estoque",
                "responsavel": "Lucas",
                "status": "Em Andamento",
                "prazo": "",
                "observacao": "Definir proposta técnica inicial.",
            },
        ]

    if "notes" not in st.session_state:
        st.session_state.notes = ""

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Olá! Sou seu agente pessoal. Posso resumir tarefas, priorizar atividades, sugerir próximos passos e montar planos de ação.",
            }
        ]


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
def build_context_text() -> str:
    tasks = st.session_state.tasks
    notes = st.session_state.notes.strip()

    lines = []
    lines.append("CONTEXTO DE TAREFAS:")
    if tasks:
        for idx, task in enumerate(tasks, start=1):
            lines.append(
                f"{idx}. Título: {task['titulo']} | Responsável: {task['responsavel']} | "
                f"Status: {task['status']} | Situação do prazo: {task_deadline_status(task)} | "
                f"Prazo: {task.get('prazo') or 'Sem prazo'} | Observação: {task.get('observacao') or 'Sem observação'}"
            )
    else:
        lines.append("Nenhuma tarefa cadastrada.")

    lines.append("")
    lines.append("ANOTAÇÕES DO USUÁRIO:")
    lines.append(notes if notes else "Sem anotações adicionais.")

    return "\n".join(lines)


# =========================
# Chamada ao modelo
# =========================
def ask_agent(user_message: str) -> str:
    client = get_client()
    model = get_secret("HF_MODEL", DEFAULT_MODEL)
    system_prompt = get_secret("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    context_text = build_context_text()

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

    for msg in st.session_state.chat_messages[-8:]:
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
                    st.session_state.tasks.append(
                        {
                            "titulo": titulo.strip(),
                            "responsavel": responsavel.strip() or "Não definido",
                            "status": status,
                            "prazo": prazo.strip(),
                            "observacao": observacao.strip(),
                        }
                    )
                    st.sidebar.success("Tarefa adicionada.")
                    st.rerun()

    with st.sidebar.expander("Anotações pessoais", expanded=True):
        notes = st.text_area(
            "Notas que o agente deve considerar",
            value=st.session_state.notes,
            height=180,
            key="notes_area",
        )
        if st.button("Salvar anotações", use_container_width=True):
            st.session_state.notes = notes
            st.sidebar.success("Anotações salvas.")

    with st.sidebar.expander("Exportar / importar contexto"):
        payload = {
            "tasks": st.session_state.tasks,
            "notes": st.session_state.notes,
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
                st.session_state.tasks = data.get("tasks", [])
                st.session_state.notes = data.get("notes", "")
                st.sidebar.success("Contexto importado com sucesso.")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Erro ao importar JSON: {exc}")

    if st.sidebar.button("Limpar conversa", use_container_width=True):
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Conversa limpa. Posso te ajudar com prioridades, tarefas e planos de ação.",
            }
        ]
        st.rerun()



def render_metrics() -> None:
    tasks = st.session_state.tasks
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

    tasks = st.session_state.tasks
    if not tasks:
        st.info("Nenhuma tarefa cadastrada.")
        return

    for idx, task in enumerate(tasks):
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
                    if st.button("Marcar como finalizada", key=f"done_{idx}", use_container_width=True):
                        st.session_state.tasks[idx]["status"] = "Finalizado"
                        st.rerun()
            with col2:
                if st.button("Excluir", key=f"delete_{idx}", use_container_width=True):
                    st.session_state.tasks.pop(idx)
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

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Pergunte algo ao seu agente pessoal")
    if not prompt and st.session_state.get("quick_prompt"):
        prompt = st.session_state.pop("quick_prompt")

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
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

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})



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
        ```

        **Observação importante**

        - Este app usa estado da sessão e importação/exportação em JSON.
        - Isso é ótimo para demo e uso pessoal leve.
        - Para persistência compartilhada entre usuários, o ideal é migrar para um banco externo.
        """
    )


# =========================
# Main
# =========================
def main() -> None:
    configure_page()
    init_session_state()
    render_sidebar()

    st.title("🤖 Agente Pessoal")
    st.caption("Assistente pessoal com tarefas, anotações e chat usando um LLM público via API.")

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
