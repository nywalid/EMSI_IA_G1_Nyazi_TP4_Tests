package ma.emsi.nyazi.Test1;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

public class RagNaif {
    public static void main(String[] args) {
        String cle= System.getenv("GEMINI_KEY");
        ChatModel modele = GoogleAiGeminiChatModel
                .builder()
                .apiKey(cle)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .build();
    }
}
