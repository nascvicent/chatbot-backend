# backend/rag_chain.py
# backend/rag_chain.py
import os
import io
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough  # <-- mudou de lugar
from langchain_core.output_parsers import StrOutputParser  # <-- mudou de lugar


# --- Ajuste: substitua CSV_DATA e exemplos_classificados pelos seus dados reais ---
CSV_DATA = """Nivel 1 (Área),Nivel 2 (Tema),Nivel 3 (Subtema)
Competência do TCE,Acesso à informação,Abrangência
Competência do TCE,Administração federal,Abrangência
Competência do TCE,Administração federal,Ato normativo
Competência do TCE,Administração federal,Controle prévio
Competência do TCE,Administração federal,Delação premiada
Competência do TCE,Administração federal,Interesse privado
Competência do TCE,Administração federal,Mediação
Competência do TCE,Administração federal,Poder discricionário
Competência do TCE,Administração federal,Recursos privados
Competência do TCE,Administração federal,Resolução consensual
Competência do TCE,Administração federal,Termo de ajustamento de conduta
Competência do TCE,Agência reguladora,Abrangência
Competência do TCE,Arrecadação da receita,Compensação financeira
Competência do TCE,Arrecadação da receita,DPVAT
Competência do TCE,Arrecadação da receita,Regime de tributação simplificada
Competência do TCE,Arrecadação da receita,Tributo
Competência do TCE,Assistência à saúde,Entidade fechada de autogestão
Competência do TCE,CGU,Prerrogativa
Competência do TCE,Chamamento público,Curso de graduação
Competência do TCE,Concurso de prognóstico,Abrangência
Competência do TCE,Concurso público,Taxa
Competência do TCE,Conselho de fiscalização profissional,Abrangência
Competência do TCE,Conselho de fiscalização profissional,Remuneração
Competência do TCE,Contrato administrativo,Abrangência
Competência do TCE,Contribuição sindical,Abrangência
Competência do TCE,Controle de constitucionalidade,Caso concreto
Competência do TCE,Convênio,Abrangência
Competência do TCE,Convênio,Adimplência
Competência do TCE,Convênio,Bens permanentes
Competência do TCE,Convênio,Organização internacional
Competência do TCE,Convênio,Prestação de contas
Competência do TCE,Convênio,Resilição
Competência do TCE,Declaração de inidoneidade,Abrangência
Competência do TCE,Denúncia,Abrangência
Competência do TCE,Desconsideração da personalidade jurídica,Abrangência
Competência do TCE,Desestatização,Abrangência
Competência do TCE,Desestatização,Ente da Federação
Competência do TCE,Desestatização,Poder discricionário
Competência do TCE,Desestatização,Relação de consumo
Competência do TCE,Despesa sigilosa,Abrangência
Competência do TCE,Determinação,Abrangência
Competência do TCE,Determinação,Natureza jurídica
Competência do TCE,Direito autoral,Ecad
Competência do TCE,Empresa estatal,Contrato internacional
Competência do TCE,Empresa estatal,Impacto ambiental
Competência do TCE,Empresa estatal,Interesse privado
Competência do TCE,Empresa estatal,Regularidade trabalhista
Competência do TCE,Empresa estatal,Responsabilidade civil
Competência do TCE,Empresa privada,Combate à corrupção
Competência do TCE,Empresa privada,Crime
Competência do TCE,Empresa supranacional,Eficácia
Competência do TCE,Empréstimo externo,Abrangência
Competência do TCE,Ente da Federação,Autonomia administrativa
Competência do TCE,Ente da Federação,Intervenção federal
Competência do TCE,Ente da Federação,Responsabilidade fiscal
Competência do TCE,Execução orçamentária,Assistência à saúde
Competência do TCE,Fundação de apoio,Fraude
Competência do TCE,Fundos,FAT
Competência do TCE,Fundos,FGTS
Competência do TCE,Fundos,FNDE
Competência do TCE,Fundos,Fundeb
Competência do TCE,Fundos,Fundo de Arrendamento Residencial
Competência do TCE,Fundos,Fundo Garantidor de Créditos
Competência do TCE,Fundos,Fundo Municipal de Saúde
Competência do TCE,Fundos,Fundo Nacional de Assistência Social
Competência do TCE,Fundos,Fundo Nacional de Saúde
Competência do TCE,Fundos,Fundo partidário
Competência do TCE,Fundos,Fundos de saúde
Competência do TCE,Fundos,Prestação de contas
Competência do TCE,Fundos,Transferências constitucionais e legais
Competência do TCE,Indisponibilidade de bens,Abrangência
Competência do TCE,Instituição financeira,Sigilo bancário
Competência do TCE,Jurisprudência,Abrangência
Competência do TCE,Licitação,Abrangência
Competência do TCE,Obra pública,Congresso Nacional
Competência do TCE,Obra pública,Impacto ambiental
Competência do TCE,Operação de crédito,Abrangência
Competência do TCE,Operação de crédito,Crédito de instalação
Competência do TCE,Pessoal,Acumulação de cargo público
Competência do TCE,Pessoal,Advogado público
Competência do TCE,Pessoal,Ato sujeito a registro
Competência do TCE,Pessoal,Conflito de interesse
Competência do TCE,Pessoal,Interesse privado
Competência do TCE,Pessoal,Pensão indenizatória
Contrato Administrativo,Aditivo,Contratação integrada
Contrato Administrativo,Aditivo,Empreitada por preço global
Contrato Administrativo,Aditivo,Limite
Contrato Administrativo,Aditivo,Requisito
Contrato Administrativo,Aditivo,Serviço novo
Contrato Administrativo,Aditivo,Sobrepreço
Contrato Administrativo,Administração contratada,Vedação
Contrato Administrativo,Anulação,Avaliação
Contrato Administrativo,Anulação,Indenização
Contrato Administrativo,Arbitragem,Requisito
Contrato Administrativo,Assistência à saúde,Medição
Contrato Administrativo,Bens e serviços de informática,Fiscalização
Contrato Administrativo,Bens e serviços de informática,Medição
Contrato Administrativo,Cessão de uso,Requisito
Contrato Administrativo,Consórcio,Composição
Contrato Administrativo,Consultoria,Fiscalização
Contrato Administrativo,Contratado,Transformação de empresa
Contrato Administrativo,Contrato de trabalho,Regularidade fiscal
Contrato Administrativo,Contrato verbal,Pagamento
Contrato Administrativo,Emergência,Vigência
Contrato Administrativo,Empreitada por preço global,Pagamento
Contrato Administrativo,Empreitada por preço global,Quantificação
Contrato Administrativo,Empreitada por preço unitário,Pagamento
Contrato Administrativo,Equilíbrio econômico-financeiro,Avaliação
Contrato Administrativo,Equilíbrio econômico-financeiro,Encargos sociais
Contrato Administrativo,Equilíbrio econômico-financeiro,Encargos trabalhistas
Contrato Administrativo,Equilíbrio econômico-financeiro,Orçamento da União
Contrato Administrativo,Equilíbrio econômico-financeiro,Orçamento estimativo
Contrato Administrativo,Equilíbrio econômico-financeiro,Preclusão lógica
Contrato Administrativo,Equilíbrio econômico-financeiro,Preço
Contrato Administrativo,Execução,Regularidade fiscal
Contrato Administrativo,Execução de contrato,Preposto
Contrato Administrativo,Execução de contrato,Regularidade fiscal
Contrato Administrativo,Fiscalização,Exigência
Contrato Administrativo,Fiscalização,Experiência profissional
Contrato Administrativo,Fiscalização,SIASG
Contrato Administrativo,Fiscalização,Terceirização
Contrato Administrativo,Formalização do contrato,Cláusula obrigatória
Contrato Administrativo,Formalização do contrato,Conteúdo
Contrato Administrativo,Formalização do contrato,Contrato verbal
Contrato Administrativo,Formalização do contrato,Depósito judicial
Contrato Administrativo,Formalização do contrato,Emergência
Contrato Administrativo,Formalização do contrato,Idioma
Contrato Administrativo,Formalização do contrato,Obrigatoriedade
Contrato Administrativo,Formalização do contrato,Seguro
Contrato Administrativo,Garantia contratual,Exigência
Contrato Administrativo,Garantia contratual,Seguro garantia
Contrato Administrativo,Liquidação da despesa,Amostra
Contrato Administrativo,Liquidação da despesa,Atestação
Contrato Administrativo,Liquidação da despesa,Nota fiscal
Contrato Administrativo,Locação de imóveis,Benfeitoria
Contrato Administrativo,Locação de imóveis,Prorrogação de contrato
Contrato Administrativo,Locação de imóveis,Sanção
Contrato Administrativo,Obras e serviços de engenharia,Aceitação provisória
Contrato Administrativo,Obras e serviços de engenharia,BDI
Contrato Administrativo,Obras e serviços de engenharia,Cronograma físico-financeiro
Contrato Administrativo,Obras e serviços de engenharia,Defeito construtivo
Contrato Administrativo,Obras e serviços de engenharia,Desapropriação de imóveis
Contrato Administrativo,Obras e serviços de engenharia,Desmobilização
Contrato Administrativo,Obras e serviços de engenharia,Encargos sociais
Contrato Administrativo,Obras e serviços de engenharia,Fiscalização
Contrato Administrativo,Obras e serviços de engenharia,Medição
Contrato Administrativo,Obras e serviços de engenharia,Mobilização
Contrato Administrativo,Obras e serviços de engenharia,Obra atrasada
Contrato Administrativo,Obras e serviços de engenharia,Obra paralisada
Contrato Administrativo,Obras e serviços de engenharia,Projeto
Contrato Administrativo,Obras e serviços de engenharia,Reajuste
Contrato Administrativo,Obras e serviços de engenharia,Recebimento
Contrato Administrativo,Obras e serviços de engenharia,Responsabilidade civil
Contrato Administrativo,Obras e serviços de engenharia,Rodovia
Contrato Administrativo,Obras e serviços de engenharia,Superfaturamento
Contrato Administrativo,Obras e serviços de engenharia,Terraplenagem
Contrato Administrativo,Pagamento,Mora
Contrato Administrativo,Pagamento antecipado,Requisito
Contrato Administrativo,Pagamento antecipado,Vedação
Contrato Administrativo,Parlamentar,Vedação
Contrato Administrativo,Preço,BDI
Contrato Administrativo,Prestação de contas,Patrocínio
Contrato Administrativo,Prestação de serviço,Regularidade fiscal
Contrato Administrativo,Princípio da publicidade,Forma
Contrato Administrativo,Projeto básico,Autor
Contrato Administrativo,Propaganda e publicidade,Bônus de volume
Contrato Administrativo,Propaganda e publicidade,Pagamento
Contrato Administrativo,Propaganda e publicidade,Patrocínio
Contrato Administrativo,Propaganda e publicidade,Subcontratação
Contrato Administrativo,Propaganda e publicidade,Vedação
Contrato Administrativo,Prorrogação de contrato,Contrato de depósito
Contrato Administrativo,Prorrogação de contrato,Inexigibilidade de licitação
Contrato Administrativo,Prorrogação de contrato,Requisito
Contrato Administrativo,Prorrogação de contrato,Serviços contínuos
Contrato Administrativo,RDC,Contratação integrada
Contrato Administrativo,Reajuste,Inadimplência
Contrato Administrativo,Reajuste,Prazo
Contrato Administrativo,Repactuação,Cabimento
Contrato Administrativo,Repactuação,Obrigatoriedade
Contrato Administrativo,Repactuação,Prazo
Contrato Administrativo,Repactuação,Preclusão lógica
Contrato Administrativo,Repactuação,Requisito
Contrato Administrativo,Rescisão amigável,Requisito
Contrato Administrativo,Rescisão unilateral,Requisito
Contrato Administrativo,Sanção administrativa,Inadimplência
Contrato Administrativo,Sanção administrativa,Obrigatoriedade
Contrato Administrativo,Seguro,Responsabilidade civil
Contrato Administrativo,Seguro,Vedação
Contrato Administrativo,Serviços advocatícios,Responsabilidade civil
Contrato Administrativo,Subcontratação,Requisito
Contrato Administrativo,Sub-rogação,Substituição
Contrato Administrativo,Sub-rogação,Vedação
Contrato Administrativo,Superfaturamento,BDI
Contrato Administrativo,Superfaturamento,Compensação financeira
Contrato Administrativo,Superfaturamento,Composição de custo unitário
Contrato Administrativo,Superfaturamento,Garantia contratual
Contrato Administrativo,Superfaturamento,Medida cautelar
Contrato Administrativo,Superfaturamento,Metodologia
Contrato Administrativo,Superfaturamento,Preço
Contrato Administrativo,Superfaturamento,Salário
Contrato Administrativo,Superfaturamento,Subcontratação
Contrato Administrativo,Taxa de administração,Vedação
Contrato Administrativo,Terceirização,Estudo de viabilidade
Contrato Administrativo,Terceirização,Fiscalização
Contrato Administrativo,Terceirização,Folha de pagamento
Contrato Administrativo,Terceirização,Inadimplência
Contrato Administrativo,Terceirização,Legislação
Contrato Administrativo,Terceirização,Medição
Contrato Administrativo,Terceirização,Pagamento
Contrato Administrativo,Terceirização,Reserva técnica
Contrato Administrativo,Terceirização,Vedação
Contrato Administrativo,Trabalho temporário,Subordinação
Contrato Administrativo,Vigência,Extinção
Contrato Administrativo,Vigência,Prazo
Convênio,Acordo de cooperação,Exército
Convênio,Acordo de cooperação,Organização internacional
Convênio,Acordo de cooperação,Órgão público
Convênio,Acordo de cooperação,Requisito
Convênio,Bens imóveis,Regularização fundiária
Convênio,Bolsa de estudo,Requisito
Convênio,Bolsa de estudo,Retorno ao Brasil
Convênio,Concedente,Licitação
Convênio,Concedente,Obrigação
Convênio,Concessão de serviço público,Obrigação
Convênio,Conselho de fiscalização profissional,Formalização
Convênio,Contrato de repasse,Execução financeira
Convênio,Convenente,Contrapartida
Convênio,Convenente,Critério de seleção
Convênio,Convenente,Inadimplência
Convênio,Convenente,Obrigação
Convênio,Convenente,Princípio da impessoalidade
Convênio,Convenente,Seleção
Convênio,Emenda parlamentar,Requisito
Convênio,Entidade fechada de autogestão,Assistência à saúde
Convênio,Entidade sem fins lucrativos,Acesso à informação
Convênio,Entidade sem fins lucrativos,Vedação
Convênio,Esporte,Financiamento
Convênio,Execução financeira,Aplicação financeira
Convênio,Execução financeira,FNDE
Convênio,Execução financeira,Nexo de causalidade
Convênio,Execução financeira,Pagamento antecipado
Convênio,Execução financeira,Receita
Convênio,Execução financeira,Sigilo bancário
Convênio,Execução financeira,Vedação
Convênio,Execução física,Desvio de objeto
Convênio,Execução física,Execução parcial
Convênio,Execução física,Medição
Convênio,Execução física,Substabelecimento
Convênio,Financiamento público,Irrigação
Convênio,Formalização,Requisito
Convênio,Fundação de apoio,Execução financeira
Convênio,Fundação de apoio,Vedação
Convênio,Legislação,Marco temporal
Convênio,Lei Rouanet,Recursos financeiros
Convênio,Lei Rouanet,Vedação
Convênio,Licitação,Aproveitamento de licitação
Convênio,Licitação,Entidade de direito privado
Convênio,Licitação,Modalidade de licitação
Convênio,Licitação,Obras e serviços de engenharia
Convênio,Licitação,Princípio da publicidade
Convênio,Organização social,Admissão de pessoal
Convênio,Organização social,Assistência à saúde
Convênio,Organização social,Contrato de gestão
Convênio,Organização social,Fiscalização
Convênio,Organização social,Legislação
Convênio,Organização social,Parentesco
Convênio,Organização social,Preço de mercado
Convênio,Organização social,Princípio da impessoalidade
Convênio,Organização social,Qualificação
Convênio,Organização social,Seleção
Convênio,Oscip,ONG
Convênio,Oscip,Qualificação técnica
Convênio,Oscip,Seleção
Convênio,Oscip,Sub-rogação
Convênio,Oscip,Termo de parceria
Convênio,Plano de trabalho,Alteração
Convênio,Plano de trabalho,Análise de custos
Convênio,Plano de trabalho,Desmembramento
Convênio,Plano de trabalho,Estudo de viabilidade
Convênio,Plano de trabalho,Objeto do convênio
Convênio,Plano de trabalho,Requisito
Convênio,Prestação de contas,Documentação
Convênio,Prestação de contas,FNDE
Convênio,Prestação de contas,Fundação de apoio
Convênio,Prestação de contas,Fundo Nacional de Assistência Social
Convênio,Prestação de contas,Fundo partidário
Convênio,Prestação de contas,Impossibilidade
Convênio,Prestação de contas,Lei Rouanet
Convênio,Prestação de contas,Patrocínio
Convênio,Prestação de contas,Programa de Garantia de Renda Mínima
Convênio,Prestação de contas,Requisito
Convênio,Prestação de contas,Tomada de contas especial
Convênio,Prestação de contas,Turismo
Convênio,Servidor público,Vedação
Convênio,Sistema S,Prestação de contas
Convênio,Sistema S,Requisito
Convênio,Sistema S,Vedação
Convênio,Subvenção econômica,Processo seletivo
Convênio,Subvenção econômica,Vedação
Convênio,Subvenção social,Vedação
Convênio,SUS,Desvio de finalidade
Convênio,SUS,Desvio de objeto
Convênio,SUS,Legislação
Convênio,SUS,Medicamento
Convênio,SUS,Saúde indígena
Convênio,SUS,Tomada de contas especial
Convênio,Terceirização,Mão de obra
"Convênio,Transferência de recursos,Conexão (Tecnologia)"
Convênio,Transferência de recursos,Intempestividade
Convênio,Transferência de recursos,Parcialidade
Convênio,Transferência de recursos,Vedação
Convênio,Vigência,Prorrogação
Desestatização,Ação de classe especial,Competência
Desestatização,Aeroporto,Plano de Exploração Aeroportuária
Desestatização,Agência reguladora,Documentação
Desestatização,Concessão pública,Alienação
Desestatização,Concessão pública,Arbitragem
Desestatização,Concessão pública,Audiência pública
Desestatização,Concessão pública,Autorização
Desestatização,Concessão pública,Bens públicos
Desestatização,Concessão pública,Caducidade da concessão
Desestatização,Concessão pública,Concessionária
Desestatização,Concessão pública,Demonstração contábil
Desestatização,Concessão pública,Estudo de viabilidade
Desestatização,Concessão pública,Investimento
Desestatização,Concessão pública,Pedágio
Desestatização,Concessão pública,Petróleo
Desestatização,Concessão pública,Preço mínimo
Desestatização,Concessão pública,Prorrogação
Desestatização,Concessão pública,Relicitação
Desestatização,Concessão pública,Restrição
Desestatização,Concessão pública,Revisão tarifária
Desestatização,Concessão pública,Sociedade de propósito específico
Desestatização,Delegação de serviço público,Ente da Federação
Desestatização,Licitação,Consórcio
Desestatização,Licitação,Contratação direta
Desestatização,Licitação,Edital de licitação
Desestatização,Licitação,Geração de energia elétrica
Desestatização,Licitação,Princípio da publicidade
Desestatização,Licitação,Projeto básico
Desestatização,Licitação,Serviço de telecomunicação
Desestatização,Parceria público-privada,Mineração
Desestatização,Petrobras,Legislação
Desestatização,Porto organizado,Arrendamento de instalação portuária
Desestatização,Porto organizado,Competência da União
Desestatização,Porto organizado,Tarifa portuária
Desestatização,Porto seco,Legislação
Desestatização,Procedimento de Manifestação de Interesse,Requisito
Desestatização,Rodovia,Meio ambiente
Desestatização,Rodovia,Pedágio
Desestatização,Rodovia,Projeto de engenharia
Desestatização,Rodovia,Revisão tarifária
Desestatização,Serviço lotérico,Legislação
Direito Processual,Acesso à informação,Advogado
Direito Processual,Acesso à informação,Legislação
Direito Processual,Acesso à informação,Ministério Público
Direito Processual,Acesso à informação,Sigilo
Direito Processual,Acórdão,Anulação
Direito Processual,Acórdão,Cumprimento
Direito Processual,Admissibilidade,Autoridade
Direito Processual,Agravo,Decisão interlocutória
Direito Processual,Agravo,Despacho de expediente
Direito Processual,Agravo,Indisponibilidade de bens
Direito Processual,Agravo,Medida cautelar
Direito Processual,Arresto,Oportunidade
Direito Processual,Arresto,Requisito
Direito Processual,Ato administrativo,Anulação
Direito Processual,Audiência,Caráter personalíssimo
Direito Processual,Audiência,Determinação
Direito Processual,Citação,Falecimento de responsável
Direito Processual,Citação,Formalização
Direito Processual,Citação,Solidariedade
Direito Processual,Citação,Validade
Direito Processual,Cobrança executiva,Bens
Direito Processual,Cobrança executiva,Requisito
Direito Processual,Coisa julgada,Auditoria
Direito Processual,Coisa julgada,Contas ordinárias
Direito Processual,Comunicação processual,Princípio da ampla defesa
Direito Processual,Comunicação processual,Validade
Direito Processual,Consulta,Admissibilidade
Direito Processual,Consulta,Interessado
Direito Processual,Contestação,Transferências constitucionais e legais
Direito Processual,Decisão monocrática,Nulidade
Direito Processual,Declaração de inidoneidade,Requisito
Direito Processual,Denúncia,Anonimato
Direito Processual,Embargos de declaração,Abrangência
Direito Processual,Embargos de declaração,Admissibilidade
Direito Processual,Embargos de declaração,Contradição
Direito Processual,Embargos de declaração,Efeito modificativo
Direito Processual,Embargos de declaração,Efeito suspensivo
Direito Processual,Embargos de declaração,Erro de fato
Direito Processual,Embargos de declaração,Erro material
Direito Processual,Embargos de declaração,Obscuridade
Direito Processual,Embargos de declaração,Omissão
Direito Processual,Embargos de declaração,Reiteração
Direito Processual,Erro de procedimento,Caracterização
Direito Processual,Incidente de uniformização de jurisprudência,Cabimento
Direito Processual,Indisponibilidade de bens,Abrangência
Direito Processual,Indisponibilidade de bens,Garantia
Direito Processual,Indisponibilidade de bens,Natureza jurídica
Direito Processual,Indisponibilidade de bens,Perda de objeto
Direito Processual,Indisponibilidade de bens,Prazo
Direito Processual,Indisponibilidade de bens,Princípio da ampla defesa
Direito Processual,Indisponibilidade de bens,Reiteração
Direito Processual,Indisponibilidade de bens,Requisito
Direito Processual,Julgamento,Antecipação
Direito Processual,Julgamento,Colegiado
Direito Processual,Julgamento,Erro material
Direito Processual,Julgamento,Fundamentação
Direito Processual,Julgamento,Notificação
Direito Processual,Julgamento,Pauta de sessão
Direito Processual,Julgamento,Processo conexo
Direito Processual,Julgamento,Suspensão
Direito Processual,Julgamento de contas,Contas extraordinárias
Direito Processual,Julgamento de contas,Contas ordinárias
Direito Processual,Julgamento de contas,Improbidade administrativa
Direito Processual,Julgamento de contas,Irregularidade continuada
Direito Processual,Julgamento de contas,Rol de responsáveis
Direito Processual,Julgamento de contas,Solidariedade
Direito Processual,Medida cautelar,Alteração
Direito Processual,Medida cautelar,Eficácia
Direito Processual,Medida cautelar,Garantia contratual
Direito Processual,Medida cautelar,Oportunidade
Direito Processual,Multa,Falecimento de responsável
Direito Processual,Multa,Pessoa jurídica
Direito Processual,Multa,Solicitação de informação
Direito Processual,Oitiva,Ministério Público junto ao TCU
Direito Processual,Oitiva,Natureza jurídica
Direito Processual,Oitiva,Unidade jurisdicionada
Direito Processual,Parte processual,Amicus curiae
Direito Processual,Parte processual,Contratado
Direito Processual,Parte processual,Denunciante
Direito Processual,Parte processual,Herdeiro
Direito Processual,Parte processual,Interessado
Direito Processual,Parte processual,Ministério Público junto ao TCU
Direito Processual,Parte processual,Órgão público
Direito Processual,Parte processual,Representante
Direito Processual,Parte processual,Sindicato
Direito Processual,Prazo,Concessão pública
Direito Processual,Prazo,Defensoria pública
Direito Processual,Prazo,Legislação
Direito Processual,Prazo,Pauta de sessão
Direito Processual,Prazo,Prorrogação
Direito Processual,Prazo,Recolhimento
Direito Processual,Prazo,Solicitação de informação
Direito Processual,Prazo,Trânsito em julgado
Direito Processual,Prestação de contas,Mora
Direito Processual,Princípio da ampla defesa,Auditoria
Direito Processual,Princípio da ampla defesa,Contas do governo
Direito Processual,Princípio da ampla defesa,Controle objetivo
Direito Processual,Princípio da ampla defesa,Defensor constituído
Direito Processual,Princípio da ampla defesa,Determinação
Direito Processual,Princípio da ampla defesa,Diligência
Direito Processual,Princípio da ampla defesa,Documentação
Direito Processual,Princípio da ampla defesa,Documento
Direito Processual,Princípio da ampla defesa,Documento novo
Direito Processual,Princípio da ampla defesa,Documento sigiloso
Direito Processual,Princípio da ampla defesa,Memorial
Direito Processual,Princípio da ampla defesa,Pedido de vista
Direito Processual,Princípio da ampla defesa,Reiteração
Direito Processual,Princípio da ampla defesa,Sustentação oral
Direito Processual,Princípio da independência das instâncias,Decisão judicial
Direito Processual,Princípio da independência das instâncias,Princípio do non bis in idem
Direito Processual,Princípio da independência das instâncias,Termo de ajustamento de conduta
Direito Processual,Princípio da presunção de veracidade,Defesa de responsável
Direito Processual,Princípio do formalismo moderado,Defesa de responsável
Direito Processual,Processo administrativo,Legislação
Direito Processual,Processo de controle externo,Custas
Direito Processual,Processo de controle externo,Legislação
Direito Processual,Processo de controle externo,Resolução consensual
Direito Processual,Processo julgado por relação,Requisito
"Direito Processual,Prova (Direito),Correio eletrônico"
"Direito Processual,Prova (Direito),Declaração"
"Direito Processual,Prova (Direito),Depoimento"
"Direito Processual,Prova (Direito),Documento fiscal"
"Direito Processual,Prova (Direito),Documento público"
"Direito Processual,Prova (Direito),Fotografia"
"Direito Processual,Prova (Direito),Indício"
"Direito Processual,Prova (Direito),Laudo"
"Direito Processual,Prova (Direito),Legislação"
"Direito Processual,Prova (Direito),Ônus da prova"
"Direito Processual,Prova (Direito),Perícia"
"Direito Processual,Prova (Direito),Prova emprestada"
"Direito Processual,Prova (Direito),Prova ilícita"
"Direito Processual,Prova (Direito),Relatório de fiscalização"
Direito Processual,Recurso,Admissibilidade
Direito Processual,Recurso,Contrarrazões
Direito Processual,Recurso,Determinação
Direito Processual,Recurso,Diligência
Direito Processual,Recurso,Efeito devolutivo
Direito Processual,Recurso,Efeito suspensivo
Direito Processual,Recurso,Falecimento de responsável
Direito Processual,Recurso,Fato novo
Direito Processual,Recurso,Ministério público junto ao TCU
Direito Processual,Recurso,Ministério Público junto ao TCU
Direito Processual,Recurso,Perda de objeto
Direito Processual,Recurso,Prazo
Direito Processual,Recurso,Preclusão consumativa
Direito Processual,Recurso,Preclusão lógica
Direito Processual,Recurso,Preclusão temporal
Direito Processual,Recurso,Princípio da boa-fé
Direito Processual,Recurso,Princípio da economia processual
Direito Processual,Recurso,Recomendação
Direito Processual,Recurso,Requisito
Direito Processual,Recurso de revisão,Admissibilidade
Direito Processual,Recurso de revisão,Documento novo
Direito Processual,Recurso de revisão,Efeito devolutivo
Direito Processual,Recurso de revisão,Efeito suspensivo
Direito Processual,Recurso de revisão,Recurso de reconsideração
Direito Processual,Regimento Interno,Legislação
Direito Processual,Relator,Competência
Direito Processual,Relator,Impedimento
Direito Processual,Representação,Admissibilidade
Direito Processual,Representação,Perda de objeto
Direito Processual,Representação,Princípio do impulso oficial
Direito Processual,Revelia,Citação por edital
Direito Processual,Revelia,Doença
Direito Processual,Revelia,Pessoa jurídica
Direito Processual,Revelia,Princípio da verdade material
Direito Processual,Revisão de ofício,Matéria de ordem pública
Direito Processual,Sobrestamento de processo,Ação judicial
Direito Processual,Sobrestamento de processo,Acordo de leniência
Direito Processual,Sobrestamento de processo,Decisão judicial
Direito Processual,Solicitação de informação,Admissibilidade
Direito Processual,Solidariedade,Credor
Direito Processual,Tomada de contas especial,Coisa julgada
Direito Processual,Tomada de contas especial,Consolidação
Direito Processual,Tomada de contas especial,Contas iliquidáveis
Direito Processual,Tomada de contas especial,Desarquivamento
Direito Processual,Tomada de contas especial,Fase interna
Direito Processual,Tomada de contas especial,Intempestividade
Direito Processual,Tomada de contas especial,Julgamento
Direito Processual,Tomada de contas especial,Limite mínimo
Direito Processual,Tomada de contas especial,Pressuposto processual
Direito Processual,Tomada de contas especial,Princípio da economia processual
Direito Processual,Tomada de contas especial,Quantificação
Direito Processual,Tomada de contas especial,Resolução consensual
Finanças Públicas,Administração pública,Disponibilidade de caixa
Finanças Públicas,Administração Pública,Disponibilidade de caixa
Finanças Públicas,Arrendamento mercantil,Bens imóveis
Finanças Públicas,Balanço patrimonial,Conta vinculada
Finanças Públicas,Balanço patrimonial,Depreciação
Finanças Públicas,BNDES,Empréstimo
Finanças Públicas,Concessionária,Inadimplência
Finanças Públicas,Concessionária,Indenização
Finanças Públicas,Conselho de fiscalização profissional,Demonstração contábil
Finanças Públicas,Conselho de fiscalização profissional,Festividade
Finanças Públicas,Conselho de fiscalização profissional,Responsabilidade fiscal
Finanças Públicas,Conta única,Concessão de uso
Finanças Públicas,Conta única,Concurso público
Finanças Públicas,Conta única,Folha de pagamento
Finanças Públicas,Conta única,Hospital universitário
Finanças Públicas,Conta única,Instituição de pesquisa
Finanças Públicas,Conta única,Instituição federal de ensino superior
Finanças Públicas,Conta única,Obrigatoriedade
Finanças Públicas,Crédito de instalação,Contabilização
Finanças Públicas,Déficit público,Princípio do equilíbrio orçamentário
Finanças Públicas,Descentralização de crédito,Termo de execução descentralizada
Finanças Públicas,Despesa pública,Empresa estatal
Finanças Públicas,Despesa pública,Festividade
Finanças Públicas,Despesa pública,Liquidação da despesa
Finanças Públicas,Despesa pública,Ressarcimento
Finanças Públicas,Despesa pública,Seguro
Finanças Públicas,Despesa pública,Sistema de Proteção Social dos Militares das Forças Armadas
Finanças Públicas,Despesa sigilosa,Caracterização
Finanças Públicas,Dívida pública,Amortização
Finanças Públicas,Dívida pública,Avaliação
Finanças Públicas,Dívida pública,Contabilização
Finanças Públicas,Dívida pública,Ente da Federação
Finanças Públicas,Emenda parlamentar,Hospital filantrópico
Finanças Públicas,Emenda parlamentar,Natureza jurídica
Finanças Públicas,Execução orçamentária,Abate-teto
Finanças Públicas,Execução orçamentária,Assistência à saúde
Finanças Públicas,Execução orçamentária,Banco Central do Brasil
Finanças Públicas,Execução orçamentária,Controle
Finanças Públicas,Execução orçamentária,Emenda parlamentar
Finanças Públicas,Execução orçamentária,Precatório
Finanças Públicas,Execução orçamentária,Requisito
Finanças Públicas,Execução orçamentária,Vedação
Finanças Públicas,Exportação,Financiamento
Finanças Públicas,Exportação,Petróleo
Finanças Públicas,FAT,Rendimento financeiro
Finanças Públicas,FCDF,Aplicação
Finanças Públicas,FCDF,Repasse
Finanças Públicas,FCDF,Tributo
Finanças Públicas,FGTS,Juros
Finanças Públicas,FGTS,Operação financeira
Finanças Públicas,Fundeb,Aplicação
Finanças Públicas,Fundo constitucional,Contabilização
Finanças Públicas,Fundo de Defesa de Direitos Difusos,Aplicação
Finanças Públicas,Fundo de Fiscalização das Telecomunicações,Desvinculação
Finanças Públicas,Fundo do Exército,Aplicação
Finanças Públicas,Fundo Nacional de Segurança Pública,Bens
Finanças Públicas,Fundo Penitenciário Nacional,Devolução
Finanças Públicas,Fust,Aplicação
Finanças Públicas,Gratificação por Encargo de Curso ou Concurso,Contabilização
Finanças Públicas,Instituição financeira,Contrato de depósito
Finanças Públicas,Instituição financeira,Garantia
Finanças Públicas,Interesse público,Ente da Federação
Finanças Públicas,Interesse público,Serviços advocatícios
Finanças Públicas,Material de construção,Contabilização
Finanças Públicas,Material de consumo,Contabilização
Finanças Públicas,Operação de crédito,Bens imóveis
Finanças Públicas,Operação de crédito,Bens móveis
Finanças Públicas,Operação de crédito,Parlamentar
Finanças Públicas,Orçamento da União,Classificação orçamentária
Finanças Públicas,Orçamento da União,Concurso público
Finanças Públicas,Orçamento da União,Crédito adicional
Finanças Públicas,Orçamento da União,Limite
Finanças Públicas,Orçamento da União,LOA
Finanças Públicas,Orçamento da União,Receita orçamentária
Finanças Públicas,Orçamento da União,Subsídio
Finanças Públicas,Ordenação de despesa,Empenho
Finanças Públicas,Ordenação de despesa,Requisito
Finanças Públicas,Pagamento indevido,Ressarcimento
Finanças Públicas,Pnae,Recursos financeiros
Finanças Públicas,Previdência complementar,Contribuição
Finanças Públicas,Programação orçamentária,Crédito orçamentário
Finanças Públicas,Pronaf,Vedação
Finanças Públicas,Receita pública,Aplicação
Finanças Públicas,Receita pública,Contabilização
Finanças Públicas,Receita pública,Desvinculação
Finanças Públicas,Receita pública,DPVAT
Finanças Públicas,Receita pública,Multa
Finanças Públicas,Receita pública,Receita corrente líquida
Finanças Públicas,Regime Próprio de Previdência Social,Aposentadoria
Finanças Públicas,Regime Próprio de Previdência Social,Auxílio-funeral
Finanças Públicas,Regime Próprio de Previdência Social,Pensão
Finanças Públicas,Renúncia de receita,Lei Rouanet
Finanças Públicas,Renúncia de receita,LOA
Finanças Públicas,Renúncia de receita,Medidas de compensação
Finanças Públicas,Renúncia de receita,Prorrogação
Finanças Públicas,Renúncia de receita,Requisito
Finanças Públicas,Responsabilidade fiscal,Despesa com pessoal
Finanças Públicas,Responsabilidade fiscal,FCDF
Finanças Públicas,Responsabilidade fiscal,Operação de crédito
Finanças Públicas,Responsabilidade fiscal,Órgão autônomo
Finanças Públicas,Responsabilidade fiscal,Projeto
Finanças Públicas,Responsabilidade fiscal,Terceirização
Finanças Públicas,Responsabilidade fiscal,Transferências voluntárias
Finanças Públicas,Restos a pagar,Cancelamento
Finanças Públicas,Restos a pagar,Diárias
Finanças Públicas,Restos a pagar,Requisito
Finanças Públicas,Restos a pagar,Vedação
Finanças Públicas,Siafi,Conformidade
Finanças Públicas,Siafi,Despesa com pessoal
Finanças Públicas,Sistema S,Despesa
Finanças Públicas,Sistema S,Receita
Finanças Públicas,Suprimento de fundos,Cartão de crédito corporativo
Finanças Públicas,Suprimento de fundos,Fracionamento da despesa
Finanças Públicas,Suprimento de fundos,Prestação de contas
Finanças Públicas,Suprimento de fundos,Requisito
Finanças Públicas,SUS,Bloco de financiamento
Finanças Públicas,SUS,Recursos financeiros
Finanças Públicas,TCU,Condenação
Finanças Públicas,Tecnologia da informação,Planejamento
Finanças Públicas,Transferência de recursos,COVID-19
Finanças Públicas,Transferência de recursos,Vedação
Finanças Públicas,Transferências constitucionais e legais,Fundo de participação
Finanças Públicas,Transferências constitucionais e legais,Vedação
Gestão Administrativa,Administração federal,Acessibilidade
Gestão Administrativa,Administração federal,Acesso à informação
Gestão Administrativa,Administração federal,Acordo de leniência
Gestão Administrativa,Administração federal,Assistência à saúde
Gestão Administrativa,Administração federal,Código de Defesa do Consumidor
Gestão Administrativa,Administração federal,Conselho de fiscalização profissional
Gestão Administrativa,Administração federal,Convenção coletiva de trabalho
Gestão Administrativa,Administração federal,Honorários advocatícios
Gestão Administrativa,Administração federal,PDV
Gestão Administrativa,Administração federal,Poder de polícia
Gestão Administrativa,Administração federal,Resolução consensual
Gestão Administrativa,Administração federal,Termo de ajustamento de conduta
"Gestão Administrativa","Agência Nacional do Petróleo, Gás Natural e Biocombustíveis","Competência"
Gestão Administrativa,Agricultura familiar,Programa de Aquisição de Alimentos
Gestão Administrativa,AGU,Competência
Gestão Administrativa,AGU,Defesa de responsável
Gestão Administrativa,AGU,Parecer jurídico
Gestão Administrativa,Alimentação escolar,Controle
Gestão Administrativa,ANA,Competência
Gestão Administrativa,ANATEL,Termo de ajustamento de conduta
Gestão Administrativa,ANEEL,Competência
Gestão Administrativa,ANS,Competência
Gestão Administrativa,Antaq,Competência
Gestão Administrativa,Ato administrativo,Princípio da publicidade
Gestão Administrativa,Ato administrativo,Validade
Gestão Administrativa,Banco do Brasil,Competência
Gestão Administrativa,Bens imóveis,Alienação
Gestão Administrativa,Bens imóveis,Locação de imóveis
Gestão Administrativa,Caixa Econômica Federal,Competência
Gestão Administrativa,CGU,Competência
Gestão Administrativa,Cgu (2003-2016),Competência
Gestão Administrativa,CNPq,Competência
Gestão Administrativa,Competição esportiva,Obrigação
Gestão Administrativa,Conselho de alimentação escolar,Composição
Gestão Administrativa,Conselho de fiscalização profissional,Acesso à informação
Gestão Administrativa,Conselho de fiscalização profissional,Competência
Gestão Administrativa,Conselho de fiscalização profissional,Tomada de contas especial
Gestão Administrativa,Controle interno,Bens móveis
Gestão Administrativa,Controle interno,Demonstração contábil
Gestão Administrativa,Controle interno,Estrutura organizacional
Gestão Administrativa,Controle interno,Gestão de risco
Gestão Administrativa,Controle interno,Imóvel funcional
Gestão Administrativa,Controle interno,Passagens
Gestão Administrativa,Controle interno,Veículo
"Gestão Administrativa,Controle interno (administração pública),Imóvel funcional"
Gestão Administrativa,Débito,Cobrança
Gestão Administrativa,Desapropriação,Competência
Gestão Administrativa,Desapropriação,Indenização
Gestão Administrativa,DNIT,Competência
Gestão Administrativa,Empresa estatal,Auditoria externa
Gestão Administrativa,Empresa estatal,Bens
Gestão Administrativa,Empresa estatal,Investimento
Gestão Administrativa,Empresa estatal,Preservação ambiental
Gestão Administrativa,Empresa estatal,Vedação
Gestão Administrativa,Empresa público-privada,Controle acionário
Gestão Administrativa,Escola pública,Taxa
Gestão Administrativa,Escola técnica,Competência
Gestão Administrativa,FNDE,Competência
Gestão Administrativa,FNE,Parlamentar
Gestão Administrativa,Funasa,Competência
Gestão Administrativa,Governança de TI,Planejamento
Gestão Administrativa,INCRA,Reforma agrária
Gestão Administrativa,Instituição federal de ensino superior,Delegação de competência
Gestão Administrativa,Meio ambiente,IBAMA
Gestão Administrativa,Meio ambiente,Impacto ambiental
Gestão Administrativa,Operação de crédito,Fiscalização
Gestão Administrativa,Órgão de controle interno,Competência
Gestão Administrativa,PAC,Mobilidade urbana
Gestão Administrativa,PAS,Inscrição
Gestão Administrativa,Poder discricionário,Cargo
Gestão Administrativa,Porto organizado,Transporte marítimo
Gestão Administrativa,Previdência complementar,Legislação
Gestão Administrativa,Programa Bolsa Família,Cadastramento
Gestão Administrativa,Programa Minha Casa Minha Vida,Competência
Gestão Administrativa,Renúncia de receita,Fundos
Gestão Administrativa,Secretaria do Tesouro Nacional,Competência
Gestão Administrativa,Serviços advocatícios,Defesa de responsável
Gestão Administrativa,Sistema S,Contabilidade
Gestão Administrativa,Sistema S,Penhora
Gestão Administrativa,Sistema S,Princípio
Gestão Administrativa,Sociedade de economia mista,Empresa subsidiária integral
Gestão Administrativa,SUS,Prestação de serviço
Gestão Administrativa,TCU,Cadirreg
Gestão Administrativa,Terceirização,Atividade-fim
Gestão Administrativa,Transferências constitucionais e legais,Coeficiente de participação
Gestão Administrativa,Transporte escolar,Requisito
"Licitação,Adjudicação,Lote (Licitação)"
Licitação,Ato administrativo,Anulação
Licitação,Ato administrativo,Revogação
Licitação,Audiência pública,Princípio da publicidade
Licitação,Audiência pública,Requisito
Licitação,Auxílio-alimentação,Edital de licitação
Licitação,Auxílio-alimentação,Especificação técnica
Licitação,Auxílio-alimentação,Rede credenciada
Licitação,Auxílio-alimentação,Taxa de administração
Licitação,BDI,Proposta de preço
Licitação,Bens e serviços de informática,Garantia contratual
Licitação,Bens e serviços de informática,Medição
Licitação,Bens e serviços de informática,Periféricos
Licitação,Bens e serviços de informática,Planejamento
Licitação,Bens e serviços de informática,Suprimento
Licitação,Bens públicos,Alienação
Licitação,Cadastramento,Sicaf
Licitação,Combustível,Rede credenciada
Licitação,Comissão de licitação,Composição
Licitação,Comissão de licitação,Princípio da segregação de funções
Licitação,Competitividade,Restrição
Licitação,Compra,Gestão de risco
Licitação,Conselho de fiscalização profissional,Alienação de bens
Licitação,Conselho de fiscalização profissional,Contratação direta
Licitação,Conselho de fiscalização profissional,Pregão
Licitação,Conselho de fiscalização profissional,Serviços advocatícios
Licitação,Consórcio,Garantia contratual
Licitação,Consórcio,Habilitação de licitante
Licitação,Consórcio,Limite máximo
Licitação,Consórcio,Poder discricionário
Licitação,Contratação direta,Inexigibilidade de licitação
Licitação,Contratação direta,Justificativa
Licitação,Contratação direta,Poder discricionário
Licitação,Contratação direta,Princípio da publicidade
Licitação,Contratação direta,Regularidade fiscal
Licitação,Convênio,Princípio da publicidade
"Licitação,Convite (Licitação),Convocação"
"Licitação,Convite (Licitação),Proposta"
Licitação,Cooperativa,Tratamento diferenciado
Licitação,Cooperativa,Vedação
Licitação,Desistência,Vedação
Licitação,Direito de preferência,Empresa coligada
Licitação,Direito de preferência,Pequena empresa
Licitação,Direito de preferência,Produto controlado
Licitação,Dispensa de licitação,Alienação
Licitação,Dispensa de licitação,Bens imóveis
Licitação,Dispensa de licitação,Capacidade técnico-operacional
Licitação,Dispensa de licitação,Concurso público
Licitação,Dispensa de licitação,Convênio
Licitação,Dispensa de licitação,ECT
Licitação,Dispensa de licitação,Emergência
Licitação,Dispensa de licitação,Empresa controlada
Licitação,Dispensa de licitação,Empresa estatal
Licitação,Dispensa de licitação,Entidade sem fins lucrativos
Licitação,Dispensa de licitação,Folha de pagamento
Licitação,Dispensa de licitação,Instituição de pesquisa
Licitação,Dispensa de licitação,Intervenção federal
Licitação,Dispensa de licitação,Licitação deserta
Licitação,Dispensa de licitação,Licitação fracassada
Licitação,Dispensa de licitação,Limite
Licitação,Dispensa de licitação,Prisão
Licitação,Dispensa de licitação,Remanescente de contrato
Licitação,Dispensa de licitação,Reputação ético-profissional
Licitação,Dispensa de licitação,Serviço de informática
Licitação,Documentação,Apresentação
Licitação,Documentação,Autenticação
Licitação,Edital de licitação,Alteração
Licitação,Edital de licitação,Anulação
Licitação,Edital de licitação,Apreciação
Licitação,Edital de licitação,Cláusula obrigatória
Licitação,Edital de licitação,Especificação técnica
Licitação,Edital de licitação,Formalização
Licitação,Edital de licitação,Impugnação
Licitação,Edital de licitação,Informação
Licitação,Edital de licitação,Marca
Licitação,Edital de licitação,Preço
Licitação,Edital de licitação,Vedação
Licitação,Edital de licitação,Veículo
Licitação,Empresa estatal,Atividade-fim
Licitação,Empresa estatal,Contratação direta
Licitação,Empresa estatal,Documentação
Licitação,Empresa estatal,Edital de licitação
Licitação,Empresa estatal,Obras e serviços de engenharia
Licitação,Empresa estatal,Orçamento estimativo
Licitação,Empresa estatal,Petrobras
Licitação,Empresa estatal,Plano de saúde
Licitação,Empresa estatal,Preço máximo
Licitação,Empresa estatal,Qualificação técnica
Licitação,Empresa estatal,Sanção administrativa
Licitação,Estudo de viabilidade,Detalhamento
"Licitação,Estudo de viabilidade,Locação (Licitação)"
Licitação,Formalização,Autos
Licitação,Fraude,Caracterização
Licitação,Habilitação de licitante,Declaração de inidoneidade
Licitação,Habilitação de licitante,Diligência
Licitação,Habilitação de licitante,Documentação
Licitação,Habilitação de licitante,Exigência
"Licitação,Habilitação de licitante,Lote (Licitação)"
Licitação,Habilitação de licitante,Objetivo
Licitação,Habilitação de licitante,Preclusão
Licitação,Habilitação de licitante,Princípio da vinculação ao instrumento convocatório
Licitação,Habilitação de licitante,Rede credenciada
Licitação,Habilitação de licitante,Vistoria
Licitação,Habilitação jurídica,Contrato social
Licitação,Habilitação jurídica,Documentação
Licitação,Inexigibilidade de licitação,Artista consagrado
Licitação,Inexigibilidade de licitação,Bens e serviços de informática
Licitação,Inexigibilidade de licitação,Credenciamento
Licitação,Inexigibilidade de licitação,Depósito judicial
Licitação,Inexigibilidade de licitação,Empresa público-privada
Licitação,Inexigibilidade de licitação,Fornecedor exclusivo
Licitação,Inexigibilidade de licitação,Princípio da isonomia
Licitação,Inexigibilidade de licitação,Serviços advocatícios
Licitação,Inexigibilidade de licitação,Serviço técnico especializado
Licitação,Julgamento,Competitividade
Licitação,Julgamento,Critério
Licitação,Julgamento,Erro material
Licitação,Julgamento,Princípio da vinculação ao instrumento convocatório
Licitação,Julgamento,Proposta técnica
Licitação,Licitação de técnica e preço,Critério
Licitação,Licitação de técnica e preço,Ponderação
Licitação,Licitação de técnica e preço,Requisito
Licitação,Licitação de técnica e preço,Serviços advocatícios
"Licitação,Licitação internacional,Convite (Licitação)"
Licitação,Licitação internacional,Edital de licitação
Licitação,Licitação internacional,Proposta
"Licitação,Locação (Licitação),Bens imóveis"
"Licitação,Locação (Licitação),Equipamentos"
"Licitação,Locação (Licitação),Evento"
Licitação,Margem de preferência,Vedação
Licitação,Nulidade,Aproveitamento
Licitação,Nulidade,Convalidação
Licitação,Nulidade,Princípio da publicidade
Licitação,Obras e serviços de engenharia,BDI
Licitação,Obras e serviços de engenharia,Fiscalização
Licitação,Obras e serviços de engenharia,Legislação
Licitação,Obras e serviços de engenharia,Licença ambiental
Licitação,Obras e serviços de engenharia,Monitoramento ambiental
Licitação,Obras e serviços de engenharia,Orçamento estimativo
Licitação,Obras e serviços de engenharia,Planejamento
Licitação,Obras e serviços de engenharia,Preço
Licitação,Obras e serviços de engenharia,Rodovia
Licitação,Orçamento estimativo,BDI
Licitação,Orçamento estimativo,Elaboração
Licitação,Orçamento estimativo,Encargos sociais
Licitação,Orçamento estimativo,Preço
Licitação,Orçamento estimativo,Sobrepreço
Licitação,Orçamento estimativo,Tributo
Licitação,Organização internacional,Legislação
Licitação,Organização social,Participação
Licitação,Oscip,Participação
Licitação,Parcelamento do objeto,Exceção
Licitação,Parcelamento do objeto,Fracionamento da despesa
"Licitação,Parcelamento do objeto,Lote (Licitação)"
Licitação,Parcelamento do objeto,Modalidade de licitação
Licitação,Parcelamento do objeto,Obrigatoriedade
Licitação,Parcelamento do objeto,Poder discricionário
Licitação,Parecer jurídico,Conteúdo
Licitação,Parecer jurídico,Embaixada
Licitação,Parecer jurídico,Princípio da segregação de funções
Licitação,Parentesco,Vedação
Licitação,Participação,Restrição
Licitação,Planejamento,Estudo de viabilidade
Licitação,Planejamento,Modalidade de licitação
Licitação,Planejamento,Previsão orçamentária
Licitação,Plano de saúde,Obrigatoriedade
Licitação,Plano de saúde,Rede credenciada
Licitação,Pregão,Amostra
Licitação,Pregão,Bens e serviços de informática
Licitação,Pregão,Direito de preferência
Licitação,Pregão,Equipe de apoio
Licitação,Pregão,Folha de pagamento
Licitação,Pregão,Garantia contratual
Licitação,Pregão,Garantia da proposta
Licitação,Pregão,Habilitação de licitante
Licitação,Pregão,Impugnação
Licitação,Pregão,Intenção de recurso
Licitação,Pregão,Lance
Licitação,Pregão,Negociação
Licitação,Pregão,Obras e serviços de engenharia
Licitação,Pregão,Obrigatoriedade
Licitação,Pregão,Orçamento estimativo
Licitação,Pregão,Possibilidade
Licitação,Pregão,Prazo
Licitação,Pregão,Pregoeiro
Licitação,Pregão,Princípio da publicidade
Licitação,Pregão,Princípio da segregação de funções
Licitação,Pregão,Proposta
Licitação,Pregão,Sanção administrativa
Licitação,Pregão eletrônico,Obrigatoriedade
Licitação,Pré-qualificação,Requisito
Licitação,Prestação de serviço,Reserva técnica
Licitação,Projeto básico,Aprovação
Licitação,Projeto básico,Autor
Licitação,Projeto básico,Detalhamento
Licitação,Projeto básico,Obras e serviços de engenharia
Licitação,Projeto básico,Planejamento
Licitação,Propaganda e publicidade,Adjudicação
Licitação,Propaganda e publicidade,Edital de licitação
Licitação,Propaganda e publicidade,Julgamento
Licitação,Proposta,Amostra
Licitação,Proposta,BDI
Licitação,Proposta,Certificação
Licitação,Proposta,Composição
Licitação,Proposta,Desclassificação
Licitação,Proposta,Encargos sociais
Licitação,Proposta,Erro material
Licitação,Proposta,Negociação
Licitação,Proposta,Participação nos lucros ou resultados
Licitação,Proposta,Pequena empresa
Licitação,Proposta,Preço
Licitação,Proposta,Validade
Licitação,Qualificação econômico-financeira,Exigência
Licitação,Qualificação econômico-financeira,Garantia da proposta
Licitação,Qualificação econômico-financeira,Índice contábil
Licitação,Qualificação técnica,Atestado de capacidade técnica
Licitação,Qualificação técnica,Certificação
Licitação,Qualificação técnica,Conselho de fiscalização profissional
Licitação,Qualificação técnica,Documentação
Licitação,Qualificação técnica,Equipamentos
Licitação,Qualificação técnica,Exigência
Licitação,Qualificação técnica,Licença ambiental
Licitação,Qualificação técnica,Prova de conceito
Licitação,RDC,Contratação integrada
Licitação,RDC,Forma
Licitação,RDC,Garantia contratual
Licitação,RDC,Intenção de recurso
Licitação,RDC,Lance
Licitação,RDC,Matriz de risco
Licitação,RDC,Obras e serviços de engenharia
Licitação,RDC,Orçamento estimativo
Licitação,RDC,Proposta técnica
Licitação,Recurso,Prazo
Licitação,Recurso,Restrição
Licitação,Regime de execução contratual,Empreitada integral
Licitação,Regime de execução contratual,Empreitada por preço global
Licitação,Regime de execução contratual,Empreitada por preço unitário
Licitação,Registro de preços,Adesão à ata de registro de preços
Licitação,Registro de preços,Adjudicação
Licitação,Registro de preços,Alteração contratual
Licitação,Registro de preços,Ata de registro de preços
Licitação,Registro de preços,Cabimento
"Licitação,Registro de preços,Lote (Licitação)"
Licitação,Registro de preços,Obras e serviços de engenharia
Licitação,Registro de preços,Parcelamento do objeto
Licitação,Registro de preços,Requisito
Licitação,Registro de preços,Vedação
Licitação,Regulamentação,Abrangência
Licitação,Regulamentação,Estrangeiro
Licitação,Sanção administrativa,Obrigatoriedade
Licitação,Sanção administrativa,Suspensão temporária
Licitação,Serviços,Medição
Licitação,Serviços contínuos,Agência de viagem
Licitação,Serviços contínuos,Reserva técnica
Licitação,Serviços contínuos,Serviço de comunicação
Licitação,Serviços contínuos,Serviço de manutenção e reparos
Licitação,Serviços contínuos,Serviço de transporte
Licitação,Serviços contínuos,Telefonia
Licitação,Serviços contínuos,Transporte escolar
Licitação,Sistema S,Adjudicação
Licitação,Sistema S,Auxílio-alimentação
Licitação,Sistema S,Controle
Licitação,Sistema S,Fiscalização
Licitação,Sistema S,Habilitação de licitante
Licitação,Sistema S,Legislação
Licitação,Sistema S,Orçamento estimativo
Licitação,Sistema S,Pregão
Licitação,Sistema S,Regularidade fiscal
Licitação,Sistema S,Tratamento diferenciado
Licitação,Sistema S,Vedação
Licitação,Sobrepreço,Crédito tributário
Licitação,Sobrepreço,Metodologia
Licitação,Terceirização,Assistência à saúde
Licitação,Terceirização,Atestado de capacidade técnica
Licitação,Terceirização,Atividade-fim
Licitação,Terceirização,Capacitação
Licitação,Terceirização,Medição
Licitação,Terceirização,Serviços advocatícios
Licitação,Veículo,Preço
Pessoal,Abandono de cargo,Caracterização
Pessoal,Abono de permanência em serviço,Requisito
Pessoal,Acumulação de cargo público,Assistente social
Pessoal,Acumulação de cargo público,Cargo eletivo
Pessoal,Acumulação de cargo público,Cargo em comissão
Pessoal,Acumulação de cargo público,Cargo técnico
Pessoal,Acumulação de cargo público,Contratação temporária
Pessoal,Acumulação de cargo público,Invalidez permanente
Pessoal,Acumulação de cargo público,Irregularidade
Pessoal,Acumulação de cargo público,Licença para tratamento de saúde
Pessoal,Acumulação de cargo público,Licença sem remuneração
Pessoal,Acumulação de cargo público,Opção
Pessoal,Acumulação de cargo público,Professor
Pessoal,Acumulação de cargo público,Profissional da área de saúde
Pessoal,Acumulação de cargo público,Proventos
Pessoal,Acumulação de cargo público,Quintos
Pessoal,Acumulação de cargo público,Regime de dedicação exclusiva
Pessoal,Acumulação de cargo público,Regime jurídico
Pessoal,Acumulação de cargo público,Servidor público militar
Pessoal,Acumulação de cargo público,Vacância do cargo
Pessoal,Acumulação de pensões,Limite
Pessoal,Acumulação de pensões,Requisito
Pessoal,Adicional de insalubridade,Requisito
Pessoal,Adicional de penosidade,Requisito
Pessoal,Adicional de periculosidade,Requisito
Pessoal,Adicional por tempo de serviço,Cálculo
Pessoal,Adicional por tempo de serviço,Empresa estatal
Pessoal,Adicional por tempo de serviço,Empresa privada
Pessoal,Adicional por tempo de serviço,Estado-membro
Pessoal,Adicional por tempo de serviço,Gratificação bienal
Pessoal,Adicional por tempo de serviço,Instituição federal de ensino
Pessoal,Adicional por tempo de serviço,Juiz classista
Pessoal,Adicional por tempo de serviço,Quinquênio
Pessoal,Adicional por tempo de serviço,Requisito
Pessoal,Adicional por tempo de serviço,Serviço militar
Pessoal,Admissão de pessoal,Contratação temporária
Pessoal,Admissão de pessoal,Fundação de apoio
Pessoal,Admissão de pessoal,Jornada de trabalho
Pessoal,Admissão de pessoal,Perícia médica
Pessoal,Admissão de pessoal,Princípio da ampla defesa
Pessoal,Admissão de pessoal,Programa Saúde da Família
Pessoal,Afastamento de pessoal,Afastamento para estudo ou missão no exterior
Pessoal,Afastamento de pessoal,Regime Próprio de Previdência Social
Pessoal,Afastamento de pessoal,Servidor público militar
Pessoal,Aposentadoria,Adicional de insalubridade
Pessoal,Aposentadoria,Adicional de periculosidade
Pessoal,Aposentadoria,Anistia
Pessoal,Aposentadoria,Anistiado político
Pessoal,Aposentadoria,Aposentadoria-prêmio
Pessoal,Aposentadoria,Cargo em comissão
Pessoal,Aposentadoria,Disponibilidade de pessoal
Pessoal,Aposentadoria,Estágio probatório
Pessoal,Aposentadoria,Idade mínima
Pessoal,Aposentadoria,Juiz classista
Pessoal,Aposentadoria,Legislação
Pessoal,Aposentadoria,Pensão civil
Pessoal,Aposentadoria,Proventos
Pessoal,Aposentadoria,Relação de emprego
Pessoal,Aposentadoria,Renúncia
Pessoal,Aposentadoria,Tabelião
Pessoal,Aposentadoria,Tempo de contribuição
Pessoal,Aposentadoria,Tempo de serviço
Pessoal,Aposentadoria,Tratado internacional
Pessoal,Aposentadoria,Vantagem opção
Pessoal,Aposentadoria,Vigência
Pessoal,Aposentadoria compulsória,Estágio probatório
Pessoal,Aposentadoria compulsória,Regime celetista
Pessoal,Aposentadoria especial,Especialista em Educação
Pessoal,Aposentadoria especial,Pessoa com deficiência
Pessoal,Aposentadoria especial,Policial
Pessoal,Aposentadoria especial,Professor
Pessoal,Aposentadoria especial,Tempo ficto
Pessoal,Aposentadoria por invalidez,Capacidade laboral
Pessoal,Aposentadoria por invalidez,Doença especificada em lei
Pessoal,Aposentadoria por invalidez,Laudo
Pessoal,Aposentadoria por invalidez,Legislação
Pessoal,Aposentadoria por invalidez,Moléstia profissional
Pessoal,Aposentadoria por invalidez,Paridade
Pessoal,Aposentadoria por invalidez,Proventos
Pessoal,Aposentadoria por invalidez,Vigência
Pessoal,Aposentadoria proporcional,Doença especificada em lei
Pessoal,Aposentadoria proporcional,Proventos
Pessoal,Ascensão funcional,Concurso público
Pessoal,Ascensão funcional,Impugnação
Pessoal,Assistência pré-escolar,Requisito
Pessoal,Ato sujeito a registro,Administração Pública
Pessoal,Ato sujeito a registro,Alteração
Pessoal,Ato sujeito a registro,Ato complexo
Pessoal,Ato sujeito a registro,Decisão judicial
Pessoal,Ato sujeito a registro,Determinação
Pessoal,Ato sujeito a registro,Inconsistência
Pessoal,Ato sujeito a registro,Julgamento de contas
Pessoal,Ato sujeito a registro,Legalidade
Pessoal,Ato sujeito a registro,Perda de objeto
Pessoal,Ato sujeito a registro,Princípio da ampla defesa
Pessoal,Ato sujeito a registro,Princípio da insignificância
Pessoal,Ato sujeito a registro,Princípio da presunção de veracidade
Pessoal,Ato sujeito a registro,Princípio da publicidade
Pessoal,Ato sujeito a registro,Princípio da razoabilidade
Pessoal,Ato sujeito a registro,Princípio da segurança jurídica
Pessoal,Ato sujeito a registro,Registro tácito
Pessoal,Ato sujeito a registro,Revisão de ofício
Pessoal,Ato sujeito a registro,Sobrestamento de processo
Pessoal,Auxílio-alimentação,Magistrado
Pessoal,Auxílio-alimentação,Requisito
Pessoal,Auxílio-funeral,Prestação de contas
Pessoal,Auxílio-funeral,Regime estatutário
Pessoal,Auxílio-invalidez,Servidor público militar
Pessoal,Auxílio-moradia,Requisito
Pessoal,Auxílio-moradia,TCU
Pessoal,Auxílio-reclusão,Acumulação
Pessoal,Auxílio-saúde,Plano de saúde
Pessoal,Avaliação de desempenho,Princípio da ampla defesa
Pessoal,Bolsa de estudo,Vedação
Pessoal,Cargo,Criação
Pessoal,Cargo em comissão,Cessão de pessoal
Pessoal,Cargo em comissão,Consultor jurídico
Pessoal,Cargo em comissão,Função de Assessoramento Superior
Pessoal,Cargo em comissão,Nepotismo
Pessoal,Cargo em comissão,Remanejamento
Pessoal,Cargo em comissão,Requisito
Pessoal,Cargo em comissão,Verba rescisória
Pessoal,Cargo público,Cargo isolado
Pessoal,Cargo público,Criação
"Pessoal,Cargo público,Investidura (pessoal)"
"Pessoal,Cargo público,Investidura (Pessoal)"
Pessoal,Cargo público,Ministério Público
Pessoal,Cargo público,Redistribuição de pessoal
Pessoal,Cargo público,Reenquadramento
"Pessoal,Cargo público,Vaga (Pessoal)"
Pessoal,Cessão de pessoal,Remuneração
Pessoal,Cessão de pessoal,Requisito
Pessoal,Colaborador eventual,Requisito
"Pessoal,Concessão (pessoal),Cancelamento"
"Pessoal,Concessão (Pessoal),Cancelamento"
Pessoal,Concurso público,Aproveitamento
Pessoal,Concurso público,Convocação
Pessoal,Concurso público,Curso de formação
Pessoal,Concurso público,Exame psicotécnico
Pessoal,Concurso público,Exigência
Pessoal,Concurso público,Experiência profissional
Pessoal,Concurso público,Pessoa com deficiência
Pessoal,Concurso público,Princípio da moralidade
Pessoal,Concurso público,Prova de títulos
Pessoal,Concurso público,Validade
Pessoal,Concurso público,Vigência
Pessoal,Conselho de fiscalização profissional,Admissão de pessoal
Pessoal,Conselho de fiscalização profissional,Bolsa de estudo
Pessoal,Conselho de fiscalização profissional,Diárias
Pessoal,Conselho de fiscalização profissional,Função de confiança
Pessoal,Conselho de fiscalização profissional,Indenização
Pessoal,Conselho de fiscalização profissional,Nepotismo
Pessoal,Conselho de fiscalização profissional,Passagens
Pessoal,Conselho de fiscalização profissional,Regime jurídico
Pessoal,Conselho de fiscalização profissional,Remuneração
Pessoal,Conselho de fiscalização profissional,Rescisão trabalhista
Pessoal,Conselho de fiscalização profissional,Serviços advocatícios
Pessoal,Conselho de fiscalização profissional,Teto constitucional
Pessoal,Curso de pós-graduação,Requisito
Pessoal,Declaração de bens e rendas,Obrigação
Pessoal,Diárias,Adicional de embarque e desembarque
Pessoal,Diárias,Desconto
Pessoal,Diárias,Prestação de contas
Pessoal,Diárias,Princípio da motivação
Pessoal,Diárias,Renúncia
Pessoal,Diárias,Vedação
Pessoal,Direitos,Falecimento
Pessoal,Edital de licitação,Cláusula obrigatória
Pessoal,Empregado público,Plano de carreira
Pessoal,Emprego público,Reenquadramento
Pessoal,Empresa estatal,Afastamento para exercício de mandato eletivo
Pessoal,Empresa estatal,Gratificação natalina
Pessoal,Empresa estatal,Nepotismo
Pessoal,Empresa estatal,Princípio da publicidade
Pessoal,Empresa estatal,Remuneração
Pessoal,Empresa pública,Cargo em comissão
Pessoal,Empresa pública,Seguro de vida
Pessoal,Empréstimo,Vedação
Pessoal,Estagiário,Nepotismo
Pessoal,Férias,Adiantamento
Pessoal,Férias,Indenização
Pessoal,Férias,Juiz classista
Pessoal,Férias,Magistrado
Pessoal,Férias,Vacância do cargo
Pessoal,Férias especiais,Irradiação ionizante
Pessoal,Função de confiança,Consultor legislativo
Pessoal,Função de confiança,Criação
Pessoal,Função de confiança,Requisito
Pessoal,Gratificação Especial de Localidade,Magistrado
Pessoal,Gratificação por Encargo de Curso ou Concurso,Vedação
Pessoal,Infração disciplinar,Prescrição
Pessoal,Infração disciplinar,Sanção administrativa
Pessoal,Jornada de trabalho,Acumulação de cargo público
Pessoal,Jornada de trabalho,Cargo em comissão
Pessoal,Jornada de trabalho,Compatibilidade de horário
Pessoal,Jornada de trabalho,Controle
Pessoal,Jornada de trabalho,Instituição federal de ensino superior
Pessoal,Jornada de trabalho,Jornada especial de trabalho
Pessoal,Jornada de trabalho,Médico
Pessoal,Jornada de trabalho,Programa Saúde da Família
Pessoal,Jornada de trabalho,Regime de sobreaviso
Pessoal,Jornada de trabalho,Teletrabalho
Pessoal,Jornada de trabalho,Trabalho noturno
Pessoal,Licença para desempenho de mandato classista,Magistrado
Pessoal,Licença para tratar de interesses particulares,Requisito
Pessoal,Licença prêmio por assiduidade,Conversão em pecúnia
Pessoal,Licença prêmio por assiduidade,Tempo residual
Pessoal,Organização internacional,Acordo de cooperação
Pessoal,Parlamentar,Aposentadoria por invalidez
Pessoal,Parlamentar,Instituto de Previdência dos Congressistas
Pessoal,Parlamentar,Verba indenizatória
Pessoal,Passivo trabalhista,Correção monetária
Pessoal,Pena disciplinar,Prescrição
Pessoal,Pensão,Anistia
Pessoal,Pensão,Base de cálculo
Pessoal,Pensão,Benefício de prestação continuada
Pessoal,Pensão,Cota-parte
Pessoal,Pensão,Limite
Pessoal,Pensão,Montepio civil
Pessoal,Pensão civil,Adiantamento pecuniário PCCS
Pessoal,Pensão civil,Capacidade laboral
Pessoal,Pensão civil,Concessão simultânea
Pessoal,Pensão civil,Cônjuge
Pessoal,Pensão civil,Dependência econômica
Pessoal,Pensão civil,Dependente designado
Pessoal,Pensão civil,Empregado público
Pessoal,Pensão civil,Filha maior solteira
Pessoal,Pensão civil,Filho adotivo
Pessoal,Pensão civil,Genitor
Pessoal,Pensão civil,Invalidez
Pessoal,Pensão civil,Juiz classista
Pessoal,Pensão civil,Legislação
Pessoal,Pensão civil,Menor sob guarda ou tutela
Pessoal,Pensão civil,Paridade
Pessoal,Pensão civil,Redutor
Pessoal,Pensão civil,Regime Próprio de Previdência Social
Pessoal,Pensão civil,União estável
Pessoal,Pensão especial,Acidente em serviço
Pessoal,Pensão especial,Viúvo
Pessoal,Pensão especial de ex-combatente,Ex-cônjuge pensionado
Pessoal,Pensão especial de ex-combatente,Filha maior solteira
Pessoal,Pensão especial de ex-combatente,Invalidez
Pessoal,Pensão especial de ex-combatente,Legislação
Pessoal,Pensão especial de ex-combatente,Vedação
Pessoal,Pensão militar,Concessão simultânea
Pessoal,Pensão militar,Cônjuge
Pessoal,Pensão militar,Dependência econômica
Pessoal,Pensão militar,Filho adotivo
Pessoal,Pensão militar,Genitor
Pessoal,Pensão militar,Invalidez
Pessoal,Pensão militar,Irmão
Pessoal,Pensão militar,Legislação
Pessoal,Pensão militar,Menor sob guarda ou tutela
Pessoal,Pensão militar,Morte ficta
Pessoal,Pensão militar,Neto
Pessoal,Pensão militar,Reforma-prêmio
Pessoal,Pensão militar,Retroatividade
Pessoal,Pensão militar,Stm
Pessoal,Pensão militar,STM
Pessoal,Pensão militar,União estável
Pessoal,Plano de carreira,Legislação
Pessoal,Policial,Legislação
Pessoal,Previdência complementar,Contribuição
Pessoal,Previdência complementar,Opção
Pessoal,Previdência complementar,Poder Legislativo
Pessoal,Processo administrativo disciplinar,Competência administrativa
Pessoal,Progressão,Promoção
Pessoal,Provimento do cargo,Readaptação de pessoal
Pessoal,Provimento do cargo,Reversão de pessoal
Pessoal,Quintos,Acumulação
Pessoal,Quintos,Adicional de gestão educacional
Pessoal,Quintos,Alteração
Pessoal,Quintos,Aposentadoria-prêmio
Pessoal,Quintos,Estado-membro
Pessoal,Quintos,Gratificação de Atividade pelo Desempenho de Função
Pessoal,Quintos,Gratificação de representação de gabinete
Pessoal,Quintos,Instituição federal de ensino
Pessoal,Quintos,Magistrado
Pessoal,Quintos,Marco temporal
Pessoal,Quintos,Requisito
Pessoal,Quintos,Secretário parlamentar
Pessoal,Quintos,Substituição de pessoal
Pessoal,Quintos,Tempo de serviço
Pessoal,Quintos,Transposição de regime jurídico
Pessoal,Quintos,Vantagem opção
Pessoal,Recondução,Enquadramento
Pessoal,Recondução,Exoneração de pessoal
"Pessoal,Reforma (pessoal),Proventos"
"Pessoal,Reforma (pessoal),Reforma-prêmio"
"Pessoal,Reforma (Pessoal),Invalidez"
"Pessoal,Reforma (Pessoal),Reforma-prêmio"
Pessoal,Regime de dedicação exclusiva,Normatização
Pessoal,Regime de dedicação exclusiva,Ressarcimento administrativo
Pessoal,Regime de dedicação exclusiva,Vedação
Pessoal,Regime Próprio de Previdência Social,Opção
Pessoal,Remoção de pessoal,Ajuda de custo
Pessoal,Remoção de pessoal,Interesse privado
Pessoal,Remoção de pessoal,Interesse público
Pessoal,Remoção de pessoal,Poder Judiciário
Pessoal,Remuneração,Adiantamento pecuniário PCCS
Pessoal,Remuneração,Adicional ocupacional
Pessoal,Remuneração,Benefício previdenciário
Pessoal,Remuneração,Coisa julgada
Pessoal,Remuneração,Decisão judicial
Pessoal,Remuneração,Direito adquirido
Pessoal,Remuneração,DPNI
Pessoal,Remuneração,Entidade sem fins lucrativos
Pessoal,Remuneração,Equiparação
Pessoal,Remuneração,Gratificação de Atividade Executiva
Pessoal,Remuneração,Gratificação de Atividade Judiciária
Pessoal,Remuneração,Gratificação de Atividade Legislativa
Pessoal,Remuneração,Gratificação de raios X
Pessoal,Remuneração,Gratificação Especial de Localidade
Pessoal,Remuneração,Gratificação por Encargo de Curso ou Concurso
Pessoal,Remuneração,Greve
Pessoal,Remuneração,Hora extra
Pessoal,Remuneração,Irredutibilidade
Pessoal,Remuneração,Licença para atividade política
Pessoal,Remuneração,Magistrado
Pessoal,Remuneração,Ministério Público
Pessoal,Remuneração,Princípio da publicidade
Pessoal,Remuneração,Professor
Pessoal,Remuneração,Regime de sobreaviso
Pessoal,Remuneração,Regime jurídico
Pessoal,Remuneração,Substituição de pessoal
Pessoal,Remuneração,Tributo
Pessoal,Remuneração,URP
Pessoal,Remuneração,URV
Pessoal,Remuneração,Vantagem pecuniária
Pessoal,Ressarcimento administrativo,Decisão judicial
Pessoal,Ressarcimento administrativo,Determinação
Pessoal,Ressarcimento administrativo,Dispensa
Pessoal,Ressarcimento administrativo,Falecimento
Pessoal,Ressarcimento administrativo,Juros de mora
Pessoal,Ressarcimento administrativo,Princípio da ampla defesa
Pessoal,Ressarcimento administrativo,Princípio da razoabilidade
Pessoal,Ressarcimento administrativo,Recurso
Pessoal,Serviço militar obrigatório,Convocação
Pessoal,Servidor público,Vedação
Pessoal,Sistema S,Admissão de pessoal
Pessoal,Sistema S,Jornada de trabalho
Pessoal,Sistema S,Nepotismo
Pessoal,Sistema S,Remuneração
Pessoal,Subsídio,Aposentadoria-prêmio
Pessoal,Subsídio,Indenização
Pessoal,Subsídio,Legislação
Pessoal,Subsídio,Quintos
Pessoal,Subsídio,Vedação
Pessoal,Telebras,ANATEL
Pessoal,Tempo de serviço,Advocacia
Pessoal,Tempo de serviço,Aluno
Pessoal,Tempo de serviço,Aluno-aprendiz
Pessoal,Tempo de serviço,Carreira
Pessoal,Tempo de serviço,Certidão pública
Pessoal,Tempo de serviço,Contagem de tempo de serviço
Pessoal,Tempo de serviço,Empregado-aprendiz
Pessoal,Tempo de serviço,Estagiário
Pessoal,Tempo de serviço,Justificação judicial
Pessoal,Tempo de serviço,Licença para tratamento de saúde
Pessoal,Tempo de serviço,Licença por motivo de doença em pessoa da família
Pessoal,Tempo de serviço,Licença prêmio por assiduidade
Pessoal,Tempo de serviço,Magistrado
Pessoal,Tempo de serviço,Mandato eletivo
Pessoal,Tempo de serviço,Menor de idade
Pessoal,Tempo de serviço,Município
Pessoal,Tempo de serviço,Pessoal militar
Pessoal,Tempo de serviço,Professor
Pessoal,Tempo de serviço,Recibado
Pessoal,Tempo de serviço,Residência médica
Pessoal,Tempo de serviço,Serviço militar
Pessoal,Tempo de serviço,Tempo de inatividade
Pessoal,Tempo de serviço,Tempo ficto
Pessoal,Tempo de serviço,Trabalho rural
Pessoal,Terceirização,Atividade-fim
Pessoal,Teto constitucional,Acumulação de cargo público
Pessoal,Teto constitucional,Base de cálculo
Pessoal,Teto constitucional,Legislação
Pessoal,Teto constitucional,Pensão
Pessoal,Titulação acadêmica,Anulação
Pessoal,Transferência de pessoal,Impugnação
Pessoal,Transposição de regime jurídico,Admissão de pessoal
Pessoal,Transposição de regime jurídico,Coisa julgada
Pessoal,Transposição de regime jurídico,Enquadramento
Pessoal,Transposição de regime jurídico,Hora extra judicial
Pessoal,Transposição de regime jurídico,Vantagem
Pessoal,Vacância do cargo,Requisito
Responsabilidade,Afastamento de responsável,Sonegação de informação
Responsabilidade,Agente político,Conduta omissiva
Responsabilidade,Agente público,Coação
Responsabilidade,Agente público,Dever de lealdade
Responsabilidade,Agente público,Formação acadêmica
Responsabilidade,Agente público,Hierarquia
Responsabilidade,Ato administrativo,Anulação
Responsabilidade,Ato administrativo,Assinatura
Responsabilidade,Ato administrativo,Magistrado
Responsabilidade,Ato administrativo,Revogação
Responsabilidade,Ato sujeito a registro,Débito
Responsabilidade,Ato sujeito a registro,Fraude
Responsabilidade,Ato sujeito a registro,Sisac
Responsabilidade,Bolsa de estudo,Débito
Responsabilidade,Bolsa de estudo,Multa
Responsabilidade,Bolsa de estudo,Obrigação
Responsabilidade,Consulta,Descumprimento
Responsabilidade,Contabilização,Fraude contábil
Responsabilidade,Contrato administrativo,Aditivo
Responsabilidade,Contrato administrativo,Agente político
Responsabilidade,Contrato administrativo,Ato antieconômico
Responsabilidade,Contrato administrativo,Conflito de interesse
Responsabilidade,Contrato administrativo,Consórcio
Responsabilidade,Contrato administrativo,Contrato verbal
Responsabilidade,Contrato administrativo,Fiscal
Responsabilidade,Contrato administrativo,Formalização
Responsabilidade,Contrato administrativo,Garantia contratual
Responsabilidade,Contrato administrativo,Licença ambiental
Responsabilidade,Contrato administrativo,Liquidação da despesa
Responsabilidade,Contrato administrativo,Medição
Responsabilidade,Contrato administrativo,Obra paralisada
Responsabilidade,Contrato administrativo,Obrigação de meio
Responsabilidade,Contrato administrativo,Parecer jurídico
Responsabilidade,Contrato administrativo,Qualidade
Responsabilidade,Contrato administrativo,Reequilíbrio econômico-financeiro
Responsabilidade,Contrato administrativo,Subcontratação
Responsabilidade,Contrato administrativo,Sub-rogação
Responsabilidade,Contrato administrativo,Superfaturamento
Responsabilidade,Contrato administrativo,Tributo
Responsabilidade,Convênio,Acordo de cooperação
Responsabilidade,Convênio,Agente político
Responsabilidade,Convênio,Concedente
Responsabilidade,Convênio,Contrapartida
Responsabilidade,Convênio,Débito
Responsabilidade,Convênio,Delegação de competência
Responsabilidade,Convênio,Desvio de finalidade
Responsabilidade,Convênio,Desvio de objeto
Responsabilidade,Convênio,Ente da Federação
Responsabilidade,Convênio,Entidade de direito privado
Responsabilidade,Convênio,Execução financeira
Responsabilidade,Convênio,Execução física
Responsabilidade,Convênio,FNDE
Responsabilidade,Convênio,Gestor sucessor
Responsabilidade,Convênio,Inabilitação de responsável
Responsabilidade,Convênio,Lei do Audiovisual
Responsabilidade,Convênio,Lei Rouanet
Responsabilidade,Convênio,Obrigação de resultado
Responsabilidade,Convênio,Omissão no dever de prestar contas
Responsabilidade,Convênio,Plano de trabalho
Responsabilidade,Convênio,Saque em espécie
Responsabilidade,Convênio,Subconvênio
Responsabilidade,Convênio,Subvenção
Responsabilidade,Crédito orçamentário,Transferência
Responsabilidade,Culpa,Conselho de administração
Responsabilidade,Culpa,Erro grosseiro
Responsabilidade,Culpa,Gestor substituto
Responsabilidade,Culpa,Parecerista
Responsabilidade,Culpa,Supervisão
Responsabilidade,Débito,Agente privado
Responsabilidade,Débito,Benefício previdenciário
Responsabilidade,Débito,Capacidade econômica
Responsabilidade,Débito,Compensação
Responsabilidade,Débito,Conduta omissiva
Responsabilidade,Débito,Correção monetária
Responsabilidade,Débito,Credor
Responsabilidade,Débito,Culpa
Responsabilidade,Débito,Desapropriação
Responsabilidade,Débito,Desconsideração da personalidade jurídica
Responsabilidade,Débito,Dolo
Responsabilidade,Débito,Estimativa
Responsabilidade,Débito,Falecimento de responsável
Responsabilidade,Débito,Fundo partidário
Responsabilidade,Débito,Imprescritibilidade
Responsabilidade,Débito,Juros de mora
Responsabilidade,Débito,Moeda estrangeira
Responsabilidade,Débito,Nexo de causalidade
Responsabilidade,Débito,Parcelamento
Responsabilidade,Débito,Penhor
Responsabilidade,Débito,Prescrição
Responsabilidade,Débito,Princípio da insignificância
Responsabilidade,Débito,Princípio do in dubio pro reo
Responsabilidade,Débito,Quitação ao responsável
Responsabilidade,Débito,Requisito
Responsabilidade,Decadência,Afastamento
Responsabilidade,Decadência,Interrupção
Responsabilidade,Decadência,Legislação
Responsabilidade,Decadência,Termo inicial
Responsabilidade,Declaração de inidoneidade,Abrangência
Responsabilidade,Declaração de inidoneidade,Acordo de leniência
Responsabilidade,Declaração de inidoneidade,Cadastro
Responsabilidade,Declaração de inidoneidade,Consórcio
Responsabilidade,Declaração de inidoneidade,Desconsideração da personalidade jurídica
Responsabilidade,Declaração de inidoneidade,Documento falso
Responsabilidade,Declaração de inidoneidade,Dosimetria
Responsabilidade,Declaração de inidoneidade,Efeito ex nunc
Responsabilidade,Declaração de inidoneidade,Garantia contratual
Responsabilidade,Declaração de inidoneidade,Inaplicabilidade
Responsabilidade,Declaração de inidoneidade,Marco temporal
Responsabilidade,Declaração de inidoneidade,Prescrição
Responsabilidade,Declaração de inidoneidade,Princípio do non bis in idem
Responsabilidade,Declaração de inidoneidade,Requisito
Responsabilidade,Declaração de inidoneidade,Sobreposição de penas
Responsabilidade,Declaração de inidoneidade,Tratamento diferenciado
Responsabilidade,Delegação de competência,Abrangência
Responsabilidade,Delegação de competência,Prestação de contas
Responsabilidade,Denunciante,Requisito
Responsabilidade,Determinação,Descumprimento
Responsabilidade,Entidade de direito privado,Conselheiro
Responsabilidade,Entidade de direito privado,Contrato administrativo
Responsabilidade,Entidade de direito privado,Contrato social
Responsabilidade,Entidade de direito privado,Empresário individual
Responsabilidade,Entidade de direito privado,Entidade filantrópica
Responsabilidade,Entidade de direito privado,Extinção
Responsabilidade,Entidade de direito privado,Princípio da boa-fé
Responsabilidade,Entidade de direito público,Revelia
Responsabilidade,Inabilitação de responsável,Abrangência
Responsabilidade,Inabilitação de responsável,Agente privado
Responsabilidade,Inabilitação de responsável,Degradação ambiental
Responsabilidade,Inabilitação de responsável,Dosimetria
Responsabilidade,Inabilitação de responsável,Fraude
Responsabilidade,Inabilitação de responsável,Individualização da pena
Responsabilidade,Inabilitação de responsável,Prescrição
Responsabilidade,Inabilitação de responsável,Princípio do non bis in idem
Responsabilidade,Inabilitação de responsável,Requisito
Responsabilidade,Inabilitação de responsável,Sobreposição de penas
Responsabilidade,Infração disciplinar,Documentação
Responsabilidade,Inimputabilidade,Requisito
Responsabilidade,Julgamento de contas,Agente privado
Responsabilidade,Julgamento de contas,Dívida
Responsabilidade,Julgamento de contas,Irregularidade
Responsabilidade,Julgamento de contas,Justiça Eleitoral
Responsabilidade,Julgamento de contas,Natureza jurídica
Responsabilidade,Julgamento de contas,Pessoa jurídica
Responsabilidade,Julgamento de contas,Prescrição
Responsabilidade,Julgamento de contas,Processo conexo
Responsabilidade,Licitação,Anulação
Responsabilidade,Licitação,Comissão de licitação
Responsabilidade,Licitação,Competitividade
Responsabilidade,Licitação,Conduta omissiva
Responsabilidade,Licitação,Contratação direta
Responsabilidade,Licitação,Dotação orçamentária
Responsabilidade,Licitação,Fraude
Responsabilidade,Licitação,Habilitação de licitante
Responsabilidade,Licitação,Homologação
Responsabilidade,Licitação,Inexigibilidade de licitação
Responsabilidade,Licitação,Medida cautelar
Responsabilidade,Licitação,Orçamento estimativo
Responsabilidade,Licitação,Parecer jurídico
Responsabilidade,Licitação,Parecer técnico
Responsabilidade,Licitação,Pregão
Responsabilidade,Licitação,Projeto básico
Responsabilidade,Licitação,Recurso
Responsabilidade,Licitação,Registro de preços
Responsabilidade,Licitação,Revogação
Responsabilidade,Licitação,Sub-rogação
Responsabilidade,Multa,Acumulação
Responsabilidade,Multa,Agente privado
Responsabilidade,Multa,Benefício previdenciário
Responsabilidade,Multa,Circunstância atenuante
Responsabilidade,Multa,Conduta omissiva
Responsabilidade,Multa,Contas ordinárias
Responsabilidade,Multa,Contas regulares com ressalva
Responsabilidade,Multa,Correção monetária
Responsabilidade,Multa,Determinação
Responsabilidade,Multa,Diligência
Responsabilidade,Multa,Dosimetria
Responsabilidade,Multa,Falecimento de responsável
Responsabilidade,Multa,Fundo partidário
Responsabilidade,Multa,Incapacidade
Responsabilidade,Multa,Litigância de má-fé
Responsabilidade,Multa,Natureza jurídica
Responsabilidade,Multa,Pagamento
Responsabilidade,Multa,Pessoa jurídica
Responsabilidade,Multa,Prescrição
Responsabilidade,Multa,Pressupostos
Responsabilidade,Multa,Revelia
Responsabilidade,Natureza jurídica,Abrangência
Responsabilidade,Nepotismo,Colaborador eventual
Responsabilidade,Nepotismo,Patrocínio
Responsabilidade,Nepotismo,Pessoa com deficiência
Responsabilidade,Obras e serviços de engenharia,Fiscalização
Responsabilidade,Obras e serviços de engenharia,Recebimento
Responsabilidade,Obras e serviços de engenharia,Superfaturamento
Responsabilidade,Ordenador de despesas,Formalização
Responsabilidade,Ordenador de despesas,Ônus da prova
Responsabilidade,Ordenador de despesas,Solidariedade
Responsabilidade,Ordenador de despesas,Supervisão
Responsabilidade,Ordenador de despesas,Transitoriedade
Responsabilidade,Organização social,Contrato de gestão
Responsabilidade,Oscip,Termo de parceria
Responsabilidade,Parecer,Supervisão
Responsabilidade,Pena disciplinar,Prescrição
Responsabilidade,Pessoal,Cessão de pessoal
Responsabilidade,Prestação de contas,Agente privado
Responsabilidade,Prestação de contas,Documentação
Responsabilidade,Prestação de contas,Fraude
Responsabilidade,Prestação de contas,Mora
Responsabilidade,Princípio da boa-fé,Avaliação
Responsabilidade,Princípio da boa-fé,Decisão judicial
Responsabilidade,Processo administrativo disciplinar,Instauração
Responsabilidade,Projeto de pesquisa,Omissão no dever de prestar contas
Responsabilidade,Recomendação,Descumprimento
Responsabilidade,Sistema S,Conselheiro
Responsabilidade,Sistema S,Débito
Responsabilidade,Sistema S,Desvio de finalidade
Responsabilidade,Sistema S,Previdência complementar
Responsabilidade,Solidariedade,Agente privado
Responsabilidade,Solidariedade,Benefício previdenciário
Responsabilidade,Solidariedade,Colegiado
Responsabilidade,Solidariedade,Conduta omissiva
Responsabilidade,Solidariedade,Credor
Responsabilidade,Solidariedade,Culpa
Responsabilidade,Solidariedade,Pagamento indevido
Responsabilidade,SUS,Agente privado
Responsabilidade,SUS,Atenção básica
Responsabilidade,SUS,Débito
Responsabilidade,SUS,Fundo Municipal de Saúde
Responsabilidade,SUS,Gestão
Responsabilidade,SUS,Medicamento
Responsabilidade,Terceirização,Atividade-fim
Responsabilidade,Tomada de contas especial,Instauração
"""  # coloque sua árvore completa em CSV

exemplos_classificados = [
    """Área:Competências do TCE | Tema:  Administração estadual e municipal | Subtema: Termo de ajustamento de conduta -- Indexadores da Ementa:  SERVIÇOS DE MANUTENÇÃO PREDIAL E DE INFRAESTRUTURA URBANA PREVENTIVA, CORRETIVA E PREDITIVA. PERICULUM IN MORA REVERSO. REVOGAÇÃO DA MEDIDA CAUTELAR. NÃO ENVIO DA PROPOSTA NO PRAZO ESTABELECIDO. IMPROCEDÊNCIA DAS ALEGAÇÕES. | Descrição: O reconhecimento, por parte deste Tribunal de Contas, de que a suspensão cautelar de procedimento licitatório que contempla a execução de serviços objeto de Termo de Ajustamento de Conduta (TAC) configura o periculum in mora reverso, não implica a permissão para que contrato decorrente da licitação eivada de vícios seja prorrogado por iguais e sucessivos períodos. | Dispositivos: 1. O reconhecimento, por parte deste
Tribunal de Contas, de que a
suspensão cautelar de procedimento
licitatório que contempla a execução
de serviços objeto de Termo de
Ajustamento de Conduta (TAC)
configura o periculum in mora
reverso, não implica a permissão
para que contrato decorrente da
licitação eivada de vícios seja
prorrogado por iguais e sucessivos
períodos.
2. Cabe ao TCE-PE, no exercício da
sua competência estabelecida no art.
2º, inciso X, da Lei nº 12.600/2004,
assinar prazo para que a edilidade
adote as providências necessárias ao
exato cumprimento da lei.


""",
    """Área: Convênio| Tema: Organização social | Subtema:Fiscalização -- Indexadores da Ementa:RECURSO ORDINÁRIO. AUDITORIA ESPECIAL. TOMADA DE CONTAS ESPECIAL. | Descrição: É contraditória a determinação de instauração de Tomada de Contas Especial e processo administrativo para desqualificação de Organização Social quando tais medidas não foram sugeridas no Parecer Ministerial integralmente acolhido. | Dispositivos: 1. Recurso Ordinário interposto
pelo Centro de Abastecimento e
Logística de Pernambuco (CEASAPE/OS) contra o Acórdão T.C. nº
413/2024, que julgou regular com
ressalvas o objeto da Auditoria
Especial TCE-PE nº 1851854-0, mas
determinou a instauração de Tomada
de Contas Especial e processo
administrativo para desqualificação
do CEASA como Organização Social.
II. QUESTÃO EM DISCUSSÃO
2. A questão em discussão consiste
em determinar se as determinações
de instauração de Tomada de Contas
Especial e de processo administrativo
para desqualificação do CEASA como
Organização Social são procedentes,
considerando o Parecer Ministerial nº
944/2022 e a ocorrência de
prescrição.
III. RAZÕES DE DECIDIR
3. O Parecer Ministerial nº
944/2022, que havia sido
integralmente acolhido pelo Tribunal,
não continha sugestão de instauração
de Tomada de Contas Especial nem
1
ESTADO DE PERNAMBUCO
TRIBUNAL DE CONTAS
de processo administrativo para
desqualificação do CEASA como
Organização Social, devido à
prescrição.
4. Mesmo que a prescrição não
fosse um fator determinante, não faz
sentido determinar a abertura de uma
Tomada de Contas Especial quando
o Tribunal não conseguiu apurar o
valor exato do débito.
IV. DISPOSITIVO E TESE
5. Recurso ordinário conhecido e
provido para afastar as
determinações de instauração de
Tomada de Contas Especial e de
processo administrativo para
desqualificação do CEASA como
Organização Social.
Teses de julgamento:
1. É contraditória a determinação
de instauração de Tomada de Contas
Especial e processo administrativo
para desqualificação de Organização
Social quando tais medidas não
foram sugeridas no Parecer
Ministerial integralmente acolhido.
2. Não se justifica a abertura de
Tomada de Contas Especial quando
o Tribunal não consegue apurar o
valor exato do débito.
Dispositivos relevantes citados: Lei
Orgânica do Tribunal de Contas do
Estado de Pernambuco, art. 73, §6º;
Resolução TC nº 36/2018, art. 13,
§2º.
Jurisprudência relevante citada: STF,
RE 636.886 AL, tema de
Repercussão Geral 899
""",
     """Área: Área: Contrato Administrativo | Tema: Pagamento antecipado | Subtema: Vedação -- Indexadores da Ementa:REPRESENTAÇÃO. MEDIDA CAUTELAR. PROCESSO LICITATÓRIO.
      SERVIÇOS DE ADMINISTRAÇÃO E INTERMEDIAÇÃO DE BENEFÍCIO DE ALIMENTAÇÃO E REFEIÇÃO. TAXA DE ADMINISTRAÇÃO. PAGAMENTO POSTERIOR
       À PRESTAÇÃO DOS SERVIÇOS. AUSÊNCIA DOS REQUISITOS PARA A CONCESSÃO | Descrição: O pagamento da taxa de administração em contrato de intermediação de benefício de auxílio-alimentação não se confunde com o próprio benefício destinado aos trabalhadores, sendo uma remuneração da empresa contratada, sujeita às regras gerais da Administração Pública. Sua exigibilidade posterior à prestação do serviço está alinhada ao art. 145 da Lei nº 14.133/2021, que veda
       adiantamentos sem justificativa e garantia de execução contratual, em observância aos princípios da economicidade e da eficiência.
        Outros Indexadores: VEDAÇÃO A ADIANTAMENTOS. JUSTIFICATIVA NECESSÁRIA | Relatório: Trata-se de apreciação por esta Primeira Câmara da decisão monocrática
proferida em 26/02/2025, publicada no DOE/TCE em 10/03/2025, por meio
da qual neguei o pedido de Medida Cautelar solicitado pela empresa MEGA
VALE ADMINISTRADORA DE CARTÕES E SERVIÇOS LTDA.
Eis o teor da decisão:
Trata-se de análise de Representação com Pedido de Medida
Cautelar (Doc. 01) protocolado pela empresa MEGA VALE
ADMINISTRADORA DE CARTÕES E SERVIÇOS
LTDA., inscrita no CNPJ sob o nº 21.922.507/0001-72, em
face de irregularidades no edital do Processo Licitatório Nº
001/2024-CPL - Licitação Eletrônica Nº 001/2024, que tem
por objeto a ‘Contratação de empresa para prestação de
serviços de administração e intermediação do benefício de
alimentação e refeição aos empregados da AGE, que
possibilitem a aquisição de gêneros alimentícios in natura e
refeições prontas através de ampla rede de estabelecimentos
credenciados, na forma definida pelos dispositivos
normativos do Ministério do Trabalho e Emprego que
regulamenta o PAT – Programa de Alimentação ao
Trabalhador, disponibilizados através de cartões com chip
de segurança, CONFORME ESPECIFICAÇÕES
DETALHADAS NO TERMO DE REFERÊNCIA, anexo ao
presente edital.’
A Representação se baseia na alegação de que o Edital viola
a Lei nº 14.442/22 ao estabelecer um prazo de pagamento de
10 dias do mês subsequente à prestação de serviço, contados
da data da emissão da nota fiscal, em vez de pagamento prépago.
A Requerente argumenta que a Lei 14.442/22, em seu artigo
3º, inciso II, determina que o pagamento do auxílioalimentação deve ser pré-pago, e que o Edital em questão
desconsidera essa exigência. A Requerente cita o
entendimento pacificado do Tribunal de Contas da União
(TCU) sobre a necessidade de pagamento antecipado do
vale-alimentação, e destaca o Acórdão 5928/2024 do TCU
como exemplo.
Ao final, a Requerente solicita a suspensão liminar do
procedimento licitatório, com a determinação de revisão do
instrumento convocatório.
Antes de decidir, conforme os termos do art. 48-B da Lei
Orgânica (Lei Estadual n º 12.600/2004) c/c art. 10 da
Resolução TC nº 155/2021, determinei a citação da
responsável, Diretora Presidente da Agência de
Empreendedorismo de Pernambuco - AGE, Sra. Angella
Mochel de Souza Netto (doc. 7).
Em síntese, a AGE argumenta que:
1) A própria Requerente (MEGA VALE
ADMINISTRADORA DE CARTÕES E SERVIÇOS
LTDA) apresentou Pedido de Esclarecimentos sobre o
mesmo objeto da Representação, questionando o prazo de
pagamento da taxa de administração.
2) O Pedido de Esclarecimentos foi formalizado em 13/02
/2025 e prontamente respondido pela Comissão Permanente
de Licitação (CPL) em 18/02/2025, mesma data em que a
Requerente distribuiu a este Tribunal sua petição com
pedido de medida cautelar, ou seja, dentro do prazo
concedido pela CPL para tal e já ciente da resposta.
3) Há um equívoco na interpretação da Requerente quanto
ao item 12.1 do instrumento convocatório. Esse item trata
do desembolso referente, se houver, ao pagamento da taxa
de administração, que deve ser realizada a título de
remuneração pela prestação do serviço, e não dos valores
destinados ao repasse de Vale-alimentação e Vale-refeição
aos funcionários da AGE. Em outras palavras, a taxa de
administração, quando aplicável, caracteriza-se como
pagamento por um serviço prestado, não se confundindo
com os valores destinados aos funcionários beneficiários
dos vales-alimentação e refeição. Desta forma, tal
remuneração poderá ser paga ao final da prestação do
serviço, conforme entendimento do TCU (Acórdão 5928
/2024, Segunda Câmara).
4) A taxa de administração e o auxílio-alimentação são
conceitos distintos. Enquanto o auxílio-alimentação é um
benefício concedido aos funcionários, a taxa de
administração refere-se à remuneração da empresa
responsável pelo gerenciamento dos cartões de valealimentação. Portanto, essa taxa não integra o benefício e
deve seguir as normas habituais de pagamento da
Administração Pública.
5) A Lei nº 14.133/2021 (Nova Lei de Licitações), em seu
artigo 141, veda o pagamento antecipado de despesas sem
justificativa e sem garantias de que o serviço foi
devidamente prestado e que a taxa administrativa somente
pode ser paga após a efetiva execução dos serviços, o que
evita desembolsos indevidos e protege o erário.
6) Caso o pagamento fosse antecipado, haveria risco de
descumprimento contratual sem a devida contraprestação, o
que contrariaria os princípios da eficiência e economicidade.
É o que importa relatar no essencia
"""
]

def formatar_arvore_como_string(csv_data):
    df = pd.read_csv(io.StringIO(csv_data))
    df.columns = ['area', 'tema', 'subtema']
    linhas_formatadas = []
    for _, row in df.iterrows():
        area = row['area'] if pd.notna(row['area']) else ""
        tema = row['tema'] if pd.notna(row['tema']) else ""
        subtema = row['subtema'] if pd.notna(row['subtema']) else ""
        linhas_formatadas.append(f"Área: {area} | Tema: {tema} | Subtema: {subtema}")
    return "\n".join(linhas_formatadas)

def build_chain():
    # Carrega API Key via env
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não encontrada nas variáveis de ambiente.")

    try:
        # IMPORTS PESADOS DYNAMICOS
        import importlib
        FAISS = importlib.import_module("langchain_community.vectorstores").FAISS
        HuggingFaceEmbeddings = importlib.import_module("langchain_community.embeddings").HuggingFaceEmbeddings
        ChatPromptTemplate = importlib.import_module("langchain_core.prompts").ChatPromptTemplate
        ChatGoogleGenerativeAI = importlib.import_module("langchain_google_genai").ChatGoogleGenerativeAI
        RunnablePassthrough = importlib.import_module("langchain.schema.runnable").RunnablePassthrough
        StrOutputParser = importlib.import_module("langchain.schema.output_parser").StrOutputParser

        print("🔹 Módulos RAG carregados dinamicamente")

        # --- Vector store ---
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = FAISS.from_texts(texts=exemplos_classificados, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # --- Prompt template ---
        opcoes_de_classificacao = formatar_arvore_como_string(CSV_DATA)
        template = """
Você é um assistente especialista em classificação de jurisprudência do Tribunal de Contas. Sua tarefa é analisar um novo documento, se inspirar nos exemplos fornecidos e, em seguida, escolher a classificação mais apropriada de uma lista de opções válidas.

---
INSPIRAÇÃO (Exemplos de documentos parecidos e suas classificações):
{context}
---
OPÇÕES VÁLIDAS (A árvore de classificação completa):
{classification_options}
---
TAREFA:
Analisando o novo documento abaixo e usando os exemplos como inspiração, ESCOLHA a classificação MAIS ADEQUADA da lista de OPÇÕES VÁLIDAS.

Novo Documento para Classificar:
{question}

Sua resposta DEVE OBRIGATORIAMENTE ser uma das opções da lista e estar no formato:
Área: [Nome da Área] | Tema: [Nome do Tema] | Subtema: [Nome do Subtema] (faça um ranking das 2 classificacoes de area tema e subtema mais possiveis)
Motivo da classificação
"""
        prompt = ChatPromptTemplate.from_template(template)

        # --- LLM ---
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True)

        # --- Fluxo híbrido RAG ---
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
                "classification_options": lambda x: opcoes_de_classificacao
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain

    except ModuleNotFoundError as e:
        print("⚠️ Dependência não encontrada:", e)
        return None
    # --- LLM ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True)

    # --- Chain híbrida (RAG) ---
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "classification_options": lambda x: opcoes_de_classificacao
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
