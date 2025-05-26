import csv
import copy
import cv2 as cv
import mediapipe as mp
import nltk
from nltk.corpus import words
from textblob import TextBlob  # << New import
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark

# Download NLTK corpus if not already downloaded
nltk.download('words')

def form_meaningful_sentence(words_list):
    """Use TextBlob to form a corrected sentence from recognized words."""
    raw_sentence = " ".join(words_list)
    blob = TextBlob(raw_sentence)
    corrected_sentence = str(blob.correct())
    return corrected_sentence.capitalize()

def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    # Load English words from nltk
    english_words = set(words.words())

    # Sentence building variables
    sentence = ""
    current_word = ""
    added_words = []
    added_words_set = set()
    prev_hand_sign_id = None
    stability_counter = 0
    stable_threshold = 5
    is_char_added = False
    invalid_char_count = 0
    max_invalid_characters = 5

    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            # Hand detected
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                if hand_sign_id == prev_hand_sign_id:
                    stability_counter += 1
                else:
                    stability_counter = 0
                    is_char_added = False

                if stability_counter >= stable_threshold and not is_char_added:
                    char = keypoint_classifier_labels[hand_sign_id]

                    if char.lower() == "space":
                        if current_word:
                            if current_word.lower() in english_words and current_word.lower() not in added_words_set:
                                added_words.append(current_word.lower())
                                added_words_set.add(current_word.lower())
                            current_word = ""
                            invalid_char_count = 0
                    elif char.lower() == "del":
                        current_word = current_word[:-1]
                        invalid_char_count = max(0, invalid_char_count - 1)
                    else:
                        current_word += char
                        if current_word.lower() in english_words:
                            if current_word.lower() not in added_words_set:
                                added_words.append(current_word.lower())
                                added_words_set.add(current_word.lower())
                            current_word = ""
                            invalid_char_count = 0
                        else:
                            invalid_char_count += 1
                            if invalid_char_count >= max_invalid_characters:
                                current_word = ""
                                invalid_char_count = 0

                    is_char_added = True
                    stability_counter = 0

                prev_hand_sign_id = hand_sign_id

                debug_image = draw_landmarks(debug_image, landmark_list)

            display_text = f"Predicted Words: {' '.join(added_words)} {current_word.capitalize()}"

        else:
            # No hand detected --> Finalize sentence
            if current_word and current_word.lower() in english_words and current_word.lower() not in added_words_set:
                added_words.append(current_word.lower())
                added_words_set.add(current_word.lower())

            if added_words:
                final_sentence = form_meaningful_sentence(added_words)
                print("Final Sentence:", final_sentence)
                with open('predicted_sentences.txt', 'a', encoding='utf-8') as f:
                    f.write(final_sentence.strip() + "\n")
            else:
                final_sentence = "No words detected."

            # Reset all
            sentence = ""
            current_word = ""
            added_words = []
            added_words_set = set()
            prev_hand_sign_id = None
            stability_counter = 0
            is_char_added = False
            invalid_char_count = 0
            display_text = f"Sentence: {final_sentence}"

        cv.putText(debug_image, display_text, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()


