# History-aware RAG Scenario Test Matrix

Use this checklist to validate `/ask` behavior from APEX or Postman.

## 1. New session, standalone question
- History: empty
- Query: `What is the LGOR2 ASN Receipt Advice integration about?`
- Expected:
  - `query_classification = standalone`
  - `winning_query` should match the current query
  - answer should be direct
  - answer should include source citation when chunks are returned

## 2. Related follow-up question
- History contains prior NPI question/answer
- Query: `show me the steps like a flow`
- Expected:
  - `query_classification = followup` or `ambiguous`
  - retrieval rewrite should include NPI context
  - answer should stay on NPI topic

## 3. Topic shift in same session
- History contains ODA discussion
- Query: `What is the LGOR2 ASN Receipt Advice integration about?`
- Expected:
  - `query_classification = standalone` or `topic_shift`
  - unrelated ODA history should not dominate retrieval
  - `winning_query` should align to LGOR2 question

## 4. Source question after a sourced answer
- History contains prior sourced assistant answer
- Query: `What is the source of this information?`
- Expected:
  - `query_classification = source_question`
  - answer should return exact available citation(s)
  - no generic no-context answer if prior context is still relevant

## 5. Duplicate current query in history
- History accidentally includes the same current user query as the last turn
- Query is also sent separately in `query`
- Expected:
  - backend sanitization removes duplicated current user query from history
  - model should not see the same user question twice

## 6. Long conversation trimming
- Send 15+ turns
- Expected:
  - backend trims to configured turn/token budget
  - recent relevant turns are preserved

## 7. Weak retrieval protection
- Use a vague query in noisy history
- Expected:
  - `relevance_passed` should indicate whether a candidate path had enough signal
  - weak candidate paths should not win too easily

## 8. No-context case
- Ask about a topic with no matching documents
- Expected:
  - no-context response
  - no fake citation
  - no meta-response like "I will search the document"

## Key response fields to inspect
- `retrieval_config.query_classification`
- `retrieval_config.retrieval_queries`
- `retrieval_config.winning_query`
- `retrieval_config.relevance_passed`
- `citations`
