import java.io.*;
import java.nio.file.Files;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Scanner;
import org.jpl7.Query;

public class NetLogoIntegrationRunner {

    public static void main(String[] args) {
        File inputFile = new File("../../netlogo_sim/jade-input.txt");
        File outputFile = new File("../../netlogo_sim/jade-output.txt");

        while (true) {
            try {
                if (inputFile.exists() && inputFile.length() > 0) {
                    String newsText = readFirstLine(inputFile);

                    if (newsText != null && !newsText.trim().isEmpty()) {
                    new PrintWriter(inputFile).close();

                    System.out.println("[NetLogo] Received for check: " + newsText);

                    boolean isFake = classify(newsText);
                    String label = isFake ? "FAKE" : "REAL";
                    String result = newsText + "|" + label;

                    Files.write(outputFile.toPath(), result.getBytes());
                    System.out.println("[NetLogo] Classified: " + result);
                    }
                }

                Thread.sleep(5000);
            } catch (Exception e) {
                System.err.println("Error in NetLogo integration:");
                e.printStackTrace();
            }
        }
    }

    private static String readFirstLine(File file) {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            return br.readLine();
        } catch (IOException e) {
            System.err.println("Error reading file:");
            e.printStackTrace();
        }
        return null;
    }

    private static boolean classify(String newsText) {
        boolean isFakeML = checkWithNLP(newsText);
        boolean isFakeSymbolic = checkWithProlog(newsText);
        return isFakeML || isFakeSymbolic;
    }

    private static boolean checkWithNLP(String newsText) {
        try {
            Thread.sleep(100 + (int) (Math.random() * 100));
            URL url = new URL("http://127.0.0.1:5000/predict");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);

            String jsonInput = "{\"text\":\"" + newsText.replace("\"", "\\\"") + "\"}";

            try (OutputStream os = conn.getOutputStream()) {
                os.write(jsonInput.getBytes("utf-8"));
                System.out.println("[NetLogo] Sent to Flask: " + jsonInput);
            }

            if (conn.getResponseCode() == 200) {
                try (Scanner scanner = new Scanner(conn.getInputStream()).useDelimiter("\\A")) {
                    String response = scanner.hasNext() ? scanner.next() : "";
                    return response.contains("\"is_fake\":true");
                }
            } else {
                System.err.println("Flask error: HTTP " + conn.getResponseCode());
            }
        } catch (Exception e) {
            System.err.println("Error in checkWithNLP:");
            e.printStackTrace();
        }
        return false;
    }

    private static boolean checkWithProlog(String newsText) {
        try {
            Query q1 = new Query("consult('../../prolog_module/fact_checker.pl')");
            q1.hasSolution();

            String cleanText = newsText.toLowerCase().replace("\"", "\\\"");
            Query q2 = new Query(String.format("fact_checker:is_fake(\"%s\")", cleanText));
            return q2.hasSolution();
        } catch (Exception e) {
            System.err.println("Error in checkWithProlog:");
            e.printStackTrace();
            return false;
        }
    }
}
