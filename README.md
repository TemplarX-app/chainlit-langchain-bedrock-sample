# Chainlit Langchain Bedrock Sample

This code will demonstate on how we can create an LLM chatbot by integrating Chainlit as UI, Langchain as the LLM framework, and Bedrock as LLM model providers (Models, Guardrails and Agents)

Replace value of following parameters with your own, you can find it in AWS Bedrock Console:
- KNOWLEDGE_BASE_ID
  - Opensearch Serverless
  - Aurora Postgres Serverless
- GUARDRAIL_ID
- GUARDRAIL_VERSION
- agent_id
- agent_alias_id

Reference:
- https://github.com/Chainlit/chainlit
- https://python.langchain.com/docs/introduction/
- https://docs.aws.amazon.com/bedrock/
