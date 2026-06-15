import { readFile } from "fs/promises";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";

import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Step - 1

// As it is intended for demos, We're using In-memory

// Database and embedding

const text = await readFile("./knowledge.txt", "utf-8")

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
  separators: ['\n\n', '\n', ' ', ''] // default setting
})

const output = await splitter.createDocuments([text]);

// The GoogleGenerativeAIEmbeddings class with the gemini-embedding-001 model in LangChain.js is restricted to the default output dimension of 3072. 
// 768 dimensions  // 3072 ?? Google recommends 3072, 1536, or 768 dimensions for quality and efficiency.

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: "AIzaSyB61kdcrD576F2m01rh8eq7PPeDAT9oI2U",
  model: "gemini-embedding-001", // 3072 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});

const vectorstore = await MemoryVectorStore.fromDocuments(
  output,
  embeddings
);

const retriever = vectorstore.asRetriever({ k: 2 });

// console.log(retriever.vectorStore.memoryVectors)

export { retriever }
