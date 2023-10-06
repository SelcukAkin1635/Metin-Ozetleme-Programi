import nltk
import rouge
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

def setup_nltk():
    
    nltk.download("stopwords")
    nltk.download("punkt")
    stop_words = set(stopwords.words("turkish"))
    return stop_words

def preprocess_text(text, stop_words):
    
    sentences = sent_tokenize(text, language="turkish")
    cleaned_sentences = [sentence for sentence in sentences if sentence.lower() not in stop_words]
    return " ".join(cleaned_sentences)

def summarize_text(text, summarizer, num_sentences=3):
   
    language = "turkish"
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    
    if summarizer == "lexrank":
        summarizer_obj = LexRankSummarizer()
    elif summarizer == "luhn":
        summarizer_obj = LuhnSummarizer()
    elif summarizer == "lsa":
        summarizer_obj = LsaSummarizer()
    else:
        summarizer_obj = TextRankSummarizer()
    
    summary = summarizer_obj(parser.document, num_sentences)
    
    return " ".join(str(sentence) for sentence in summary)

def evaluate_rouge(reference, summary):
    # Rouge metriğiyle özetleme sonuçlarını değerlendirelim
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=True,
                            alpha=0.5, # F1 puanının ağırlığı
                            weight_factor=1.2,
                            stemming=True)
    
    scores = evaluator.get_scores(summary, reference)
    return scores

if __name__ == "__main__":
    print("Metin Özetleme Programı")
    print("-----------------------")
    
   
    stop_words = setup_nltk()
    
 
    choice = input("Metni doğrudan girmek için '1', metin dosyası kullanmak için '2' girin: ")
    if choice == '1':
        input_text = input("Özetlemek istediğiniz metni girin:\n")
    elif choice == '2':
        file_path = input("Metin dosyasının yolunu girin: ")
        with open(file_path, "r", encoding="utf-8") as file:
            input_text = file.read()
    else:
        print("Geçersiz seçim. Program sonlandırılıyor.")
        exit(1)
    

    print("\nKullanılabilir Özetleme Yöntemleri:")
    print("1. LexRank")
    print("2. Luhn")
    print("3. LSA")
    print("4. TextRank")
    summarizer_choice = input("Özetleme yöntemi seçin (1/2/3/4): ")
    

    num_sentences_to_summarize = int(input("Özetde kaç cümle kullanılsın?: "))
    
   
    cleaned_text = preprocess_text(input_text, stop_words)
    
    if summarizer_choice == "1":
        summarizer_name = "LexRank"
    elif summarizer_choice == "2":
        summarizer_name = "Luhn"
    elif summarizer_choice == "3":
        summarizer_name = "LSA"
    else:
        summarizer_name = "TextRank"
    
    summarized_text = summarize_text(cleaned_text, summarizer_choice, num_sentences_to_summarize)
    
    
    print("\n{} ile Özetlenmiş Metin ({} cümle):".format(summarizer_name, num_sentences_to_summarize))
    print(summarized_text)
    
    
    reference_summary = input("\nMetnin gerçek özetlemesini girin (referans olarak kullanılacak):\n")
    
   
    rouge_scores = evaluate_rouge(reference_summary, summarized_text)
    print("\nRouge Metrikleri:")
    for metric, score in rouge_scores.items():
        print(f"{metric}:")
        for measure, value in score.items():
            print(f"    {measure}: {value:.3f}")
