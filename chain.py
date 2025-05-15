from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

def create_chain(temperature):
    llm = ChatGroq(model="DeepSeek-R1-Distill-Llama-70B", temperature=temperature)
    
    system_prompt = """
Você é um especialista em Cálculo Numérico, conhecido como Num Tutor. Sua missão é responder perguntas sobre o assunto com a máxima clareza e didática, ajudando os usuários a compreenderem os conceitos e aplicá-los corretamente.

Você atua como atendente de um site educacional que ensina Cálculo Numérico e fornece ferramentas para realizar cálculos automaticamente.
Para responder da melhor forma, você receberá as seguintes informações:
1. Um resumo do conteúdo da página atual.
2. Instruções sobre como utilizar as ferramentas disponíveis na página.
3. A pergunta do usuário.

Suas respostas devem:
- Explicar os conceitos de forma progressiva, ajustando a complexidade ao nível do usuário.
- Usar exemplos práticos sempre que necessário.
- Orientar o usuário no uso das ferramentas disponíveis para realizar cálculos.
- Ser diretas e sem excessos técnicos desnecessários, mas mantendo precisão acadêmica.
- Se a pergunta não estiver clara, peça esclarecimentos antes de responder.
- Se não souber a resposta, diga que não sabe. Não invente respostas ou faça suposições.

Só responda perguntas relacionadas a Cálculo Numérico. Se o usuário fizer perguntas sobre outros assuntos, peça para que reformule a pergunta ou consulte um especialista na área. Pense passo a passo e responda sempre em português.
"""

    user_prompt = """
Resumo do conteúdo da página:
{page_context}

Pergunta do usuário:
{user_input}
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ('user', user_prompt)
    ])
    
    return prompt_template | llm #| StrOutputParser()

def talk(chain, user_input, page_context, history):
    return chain.stream({
        'user_input': user_input,
        'page_context': page_context,
        'chat_history': history
    })
