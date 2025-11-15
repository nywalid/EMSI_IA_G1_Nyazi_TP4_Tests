package ma.emsi.nyazi.Test4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.nyazi.Test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class RagNaif4 {
        public static void main(String[] args) {

            class QueryRouterPourEviterRag implements QueryRouter {

                private final ChatModel modele;
                private final ContentRetriever contentRetriever;

                public QueryRouterPourEviterRag(ChatModel modele, ContentRetriever contentRetriever) {
                    this.modele = modele;
                    this.contentRetriever = contentRetriever;
                }

                @Override
                public List<ContentRetriever> route(Query query) {

                    String question =
                            "Est-ce que la requête '" + query.text()
                                    + "' porte sur l'IA ? Réponds seulement par"
                                    + " 'oui', 'non', ou 'peut-être'.";

                    String reponse = modele.chat(question);

                    System.out.println(">>> Décision du LM : " + reponse);

                    if (reponse.toLowerCase().contains("non")) {
                        return Collections.emptyList();
                    } else {
                        return Collections.singletonList(contentRetriever);
                    }
                }
            }
            String cle = System.getenv("GEMINI_KEY");
            ChatModel modele = GoogleAiGeminiChatModel
                    .builder()
                    .apiKey(cle)
                    .modelName("gemini-2.5-flash")
                    .temperature(0.7)
                    .logRequestsAndResponses(true)
                    .build();

            Path pdf = Paths.get("src/main/resources/rag.pdf");
            DocumentParser documentParser = new ApacheTikaDocumentParser();
            Document document = FileSystemDocumentLoader.loadDocument(pdf, documentParser);

            DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);
            List<TextSegment> segments = splitter.split(document);

            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
            Response<List<Embedding>> response = embeddingModel.embedAll(segments);
            List<Embedding> listeEmbedding = response.content();

            EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
            embeddingStore.addAll(listeEmbedding, segments);

            ContentRetriever contentRetriever =
                    EmbeddingStoreContentRetriever.builder()
                            .embeddingStore(embeddingStore)
                            .embeddingModel(embeddingModel)
                            .maxResults(3)
                            .minScore(0.5)
                            .build();

            QueryRouter router = new QueryRouterPourEviterRag(modele, contentRetriever);

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
