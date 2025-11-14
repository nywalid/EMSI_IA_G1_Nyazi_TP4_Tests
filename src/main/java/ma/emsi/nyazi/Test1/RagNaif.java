package ma.emsi.nyazi.Test1;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class RagNaif {
    public static void main(String[] args) {
        String cle= System.getenv("GEMINI_KEY");
        ChatModel modele = GoogleAiGeminiChatModel
                .builder()
                .apiKey(cle)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .build();

        Path pdf = Paths.get("src/main/resources/rag.pdf");
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(pdf,documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);

        List<TextSegment> segments = splitter.split(document);


        EmbeddingModel embeddingModel= new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> response= embeddingModel.embedAll(segments);
        List<Embedding> listeEmbedding= response.content();

        EmbeddingStore<TextSegment> embeddingStore= new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(listeEmbedding, segments);
    }
}
