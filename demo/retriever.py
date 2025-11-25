import sys
sys.dont_write_bytecode = True

from typing import List, Literal
from pydantic import BaseModel, Field

from langchain.prompts import ChatPromptTemplate


RAG_K_THRESHOLD = 5


class ApplicantID(BaseModel):
  """
  List of IDs of the applicants to retrieve resumes for
  """
  id_list: List[str] = Field(..., description="List of IDs of the applicants to retrieve resumes for")

class JobDescription(BaseModel):
  """
  Descriptions of a job to retrieve similar resumes for
  """
  job_description: str = Field(..., description="Descriptions of a job to retrieve similar resumes for")

class QueryIntent(BaseModel):
  """
  Determines the intent of the user query
  """
  query_type: Literal["retrieve_applicant_id", "retrieve_applicant_jd", "no_retrieve"] = Field(..., description="The type of query")
  extracted_input: str = Field(..., description="The extracted input for the query (ID list, job description, or empty)") 



class RAGRetriever():
  def __init__(self, vectorstore_db, df):
    self.vectorstore = vectorstore_db
    self.df = df

  def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
    fused_scores = {}
    for doc_list in document_rank_list:
      for rank, (doc, _) in enumerate(doc_list.items()):
        if doc not in fused_scores:
          fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

  def __retrieve_docs_id__(self, question: str, k=50):
    docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
    docs_score = {str(doc.metadata["ID"]): score for doc, score in docs_score}
    return docs_score

  def retrieve_id_and_rerank(self, subquestion_list: list):
    document_rank_list = []
    for subquestion in subquestion_list:
      document_rank_list.append(self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD))
    reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
    return reranked_documents

  def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
    id_resume_dict = dict(zip(self.df["ID"].astype(str), self.df["Resume"]))
    retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True))[:threshold]
    retrieved_documents = [id_resume_dict[id] for id in retrieved_ids]
    for i in range(len(retrieved_documents)):
      retrieved_documents[i] = "Applicant ID " + retrieved_ids[i] + "\n" + retrieved_documents[i]
    return retrieved_documents 
   


class SelfQueryRetriever(RAGRetriever):
  def __init__(self, vectorstore_db, df):
    super().__init__(vectorstore_db, df)

    self.prompt = ChatPromptTemplate.from_messages([
      ("system", "You are an expert in talent acquisition."),
      ("user", "{input}")
    ])
    self.meta_data = {
      "rag_mode": "",
      "query_type": "no_retrieve",
      "extracted_input": "",
      "subquestion_list": [],
      "retrieved_docs_with_scores": []
    }

  def retrieve_docs(self, question: str, llm, rag_mode: str):
    def retrieve_applicant_id(id_list: list):
      """Retrieve resumes for applicants in the id_list"""
      retrieved_resumes = []

      for id in id_list:
        try:
          resume_df = self.df[self.df["ID"].astype(str) == id].iloc[0][["ID", "Resume"]]
          resume_with_id = "Applicant ID " + resume_df["ID"].astype(str) + "\n" + resume_df["Resume"]
          retrieved_resumes.append(resume_with_id)
        except:
          return []
      return retrieved_resumes

    def retrieve_applicant_jd(job_description: str):
      """Retrieve similar resumes given a job description"""
      subquestion_list = [job_description]

      if rag_mode == "RAG Fusion":
        subquestion_list += llm.generate_subquestions(question)
        
      self.meta_data["subquestion_list"] = subquestion_list
      retrieved_ids = self.retrieve_id_and_rerank(subquestion_list)
      self.meta_data["retrieved_docs_with_scores"] = retrieved_ids
      retrieved_resumes = self.retrieve_documents_with_id(retrieved_ids)
      return retrieved_resumes
    
    self.meta_data["rag_mode"] = rag_mode
    
    # Use structured output to determine query intent
    structured_llm = llm.llm.with_structured_output(QueryIntent)
    
    prompt_with_instructions = ChatPromptTemplate.from_messages([
      ("system", """You are an expert in talent acquisition. Analyze the user query and determine:
1. If the query contains applicant IDs (like "ID: 123" or "applicant 456"), set query_type to "retrieve_applicant_id" and extract the IDs into extracted_input as a JSON list.
2. If the query contains a job description or requirements, set query_type to "retrieve_applicant_jd" and put the job description in extracted_input.
3. If the query is a general question or follow-up that doesn't require retrieval, set query_type to "no_retrieve" and leave extracted_input empty.

Return your analysis as JSON with query_type and extracted_input fields."""),
      ("user", "{input}")
    ])
    
    chain = prompt_with_instructions | structured_llm
    intent_result = chain.invoke({"input": question})
    
    self.meta_data["query_type"] = intent_result.query_type
    self.meta_data["extracted_input"] = intent_result.extracted_input
    
    if intent_result.query_type == "retrieve_applicant_id":
      import json
      try:
        id_list = json.loads(intent_result.extracted_input) if isinstance(intent_result.extracted_input, str) else intent_result.extracted_input
        result = retrieve_applicant_id(id_list)
      except:
        result = []
    elif intent_result.query_type == "retrieve_applicant_jd":
      result = retrieve_applicant_jd(intent_result.extracted_input)
    else:
      result = []

    return result
