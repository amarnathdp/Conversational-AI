import express from "express";
import dotenv from "dotenv";
import cors from "cors";

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { retriever } from './utils/retriever.js'
import { combineDocuments } from './utils/combineDocuments.js'
import { formatConvHistory } from './utils/formatConvHistory.js'

import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";

dotenv.config();
const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static("public"));

const llm = new ChatGoogleGenerativeAI({
    apiKey: "AIzaSyACT8KjqrkOD0HGt_eqLgkzK-PTaz_ZsKQ",
    modelName: "gemini-1.5-flash",
});

const convHistory = []

// Prompts
// const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question.
// If you don't know the answer, say "I'm sorry, I don't know the answer to that."
// Always speak as if you were chatting with a friend. Question: {question} Answer:`;

// const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrim.
// Try to find the answer in the context. If you don't know the answer, say "I'm sorry, I don't know the answer to that."
// Always speak as if you were chatting with a friend. Question: {question} Answer:`;

// const answerPrompt = ChatPromptTemplate.fromTemplate(answerTemplate);

app.post("/chat", async (req, res) => {
    try {
        const { question } = req.body;
        if (!question) {
            return res.status(400).json({ error: "Question is required" });
        }
        console.log(question);

        const llm = new ChatGoogleGenerativeAI({
            apiKey: "",
            modelName: "gemini-1.5-flash",
            // safetySettings
        });

        // A string holding the phrasing of the prompt
        // const standaloneQuestionTemplate = 'Given a question, convert it to a standalone question. question: {question} standalone question:'

        const standaloneQuestionTemplate = `Given some conversation history (if any) and a question, convert the question to a standalone question. 
        conversation history: {conv_history}
        question: {question} 
        standalone question:`

        // A prompt created using PromptTemplate and the fromTemplate method
        const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(standaloneQuestionTemplate)

        // const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. 
        //   Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. 
        //   Always speak as if you were chatting to a friend.
        //   context: {context}
        //   question: {question}
        //   answer: 
        //   `

        // With Conversation History
        const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about LTIAcademy based on the context provided and the conversation history.
        Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible. If you really don't know the answer, say "I'm sorry, I don't know the answer to that."
        And direct the questioner to email amarnathdp17@gmail.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
        context: {context}
        conversation history: {conv_history}
        question: {question}
        answer: `
        const answerPrompt = ChatPromptTemplate.fromTemplate(answerTemplate)

        const standaloneQuestionChain = standaloneQuestionPrompt
            .pipe(llm)
            .pipe(new StringOutputParser())

        const retrieverChain = RunnableSequence.from([
            prevResult => prevResult.standalone_question,
            retriever,
            combineDocuments
        ])
        const answerChain = answerPrompt
            .pipe(llm)
            .pipe(new StringOutputParser())

        const chain = RunnableSequence.from([
            {
                standalone_question: standaloneQuestionChain,
                original_input: new RunnablePassthrough()
            },
            {
                context: retrieverChain,
                question: ({ original_input }) => original_input.question,
                conv_history: ({ original_input }) => original_input.conv_history
            },
            answerChain
        ])

        const response = await chain.invoke({
            question: question,
            conv_history: formatConvHistory(convHistory)
        })

        convHistory.push(question)
        convHistory.push(response)

        // console.log(convHistory);
        

        // const response = await answerPrompt.pipe(llm).pipe(new StringOutputParser()).invoke({ question });
        res.json({ answer: response });
    } catch (error) {
        console.error("Error:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});