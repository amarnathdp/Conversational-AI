import { readFile } from "fs/promises";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";

import { MemoryVectorStore } from "langchain/vectorstores/memory";

// As it is intended for demos, We're using In-memory

// Database and embedding

const text = await readFile("./knowledge.txt", "utf-8")

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
  separators: ['\n\n', '\n', ' ', ''] // default setting
})

const output = await splitter.createDocuments([text])

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: "AIzaSyACT8KjqrkOD0HGt_eqLgkzK-PTaz_ZsKQ",
  model: "embedding-001", // 768 dimensions
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