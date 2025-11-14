package ma.emsi.nyazi.Test3;


import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.nyazi.Test1.Assistant;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;


public class TestRoutage {
    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
    private static EmbeddingStore<TextSegment> creerEmbeddingStore(String cheminFichier) {
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(Paths.get(cheminFichier),documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);

        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel= new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> response= embeddingModel.embedAll(segments);
        List<Embedding> listeEmbedding= response.content();

        EmbeddingStore<TextSegment> embeddingStore= new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(listeEmbedding, segments);

        return embeddingStore;
    }

    public static void main(String[] args){
        String cle= System.getenv("GEMINI_KEY");
        ChatModel modele = GoogleAiGeminiChatModel
                .builder()
                .apiKey(cle)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .build();

        EmbeddingStore<TextSegment> storeRAG = creerEmbeddingStore("src/main/resources/rag.pdf");
        EmbeddingStore<TextSegment> storeEMSI = creerEmbeddingStore("src/main/resources/Brochure_EMSI.pdf");


        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        ContentRetriever retrieverRAG = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeRAG)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        ContentRetriever retrieverEMSI = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeEMSI)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverRAG, "Documents techniques sur l'intelligence artificielle, RAG, embeddings, modèles généraux.");
        descriptions.put(retrieverEMSI, "Document de EMSI, formations et parcours");


        QueryRouter router = new LanguageModelQueryRouter(modele, descriptions);

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(modele)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }

    }
}
