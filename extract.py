import re
import os

# The directory where your text files are stored
directory = '/home/donald.peltier/swarm/logs/noise/1'  # Replace with the path to your files
model ='TRS'

# Patterns to match the desired numbers
val_loss_pattern = r"Minimum Val Loss: ([0-9.]+)"
metrics_pattern = r"model.metrics_names:\s*\[.*?\]\n\[(.*?)\]"
f1_score_pattern = r"(\bCOMMS\b|\bPRONAV\b)\s+\d\.\d+\s+\d\.\d+\s+(\d\.\d+)"

# Lists to store the extracted numbers
val_losses = []
attr_acc = []
class_acc = []
comms_f1_scores = []
pronav_f1_scores = []

# Loop over the files in the directory
for i in range(1, 51):
    filename = f"swarm-class_{model}noise{i}.txt"
    filepath = os.path.join(directory, filename)

    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            content = file.read()

            # Extract 'Minimum Val Loss'
            val_loss_matches = re.findall(val_loss_pattern, content)
            if val_loss_matches:
                val_loss = int(round(float(val_loss_matches[0])*100))
                val_losses.append(val_loss)
            else:
                val_losses.append('N/A')

            # Extract accuracy values
            metrics_matches = re.findall(metrics_pattern, content, re.MULTILINE)
            for match in metrics_matches:
                numbers = [x.strip() for x in match.split(',')]
                if len(numbers) >= 5:
                    class_acc.append(int(round(float(numbers[3])*100)))  # 4th number
                    attr_acc.append(int(round(float(numbers[4])*100)))  # 5th number
                else:
                    class_acc.append('N/A')
                    attr_acc.append('N/A')
            
            # Extract f1-scores for COMMS and PRONAV
            f1_scores = re.findall(f1_score_pattern, content)
            # Initialize to 'N/A' which will remain if not found
            comms_f1 = 'N/A'
            pronav_f1 = 'N/A'
            for match in f1_scores:
                if match[0] == 'COMMS':
                    comms_f1 = int(round(float(match[1])*100))
                elif match[0] == 'PRONAV':
                    pronav_f1 = int(round(float(match[1])*100))
            comms_f1_scores.append(comms_f1)
            pronav_f1_scores.append(pronav_f1)

    else:
        val_losses.append('N/A')
        class_acc.append('N/A')
        attr_acc.append('N/A')
        print(f"File not found: {filename}")

# The directory where you want to save the extracted_data.txt file
output_directory = '/home/donald.peltier/swarm/logs/extracted_data'  # Replace with the path where you want to save the file
out_filename = f'extracted_data_{model}.txt'
output_filename = os.path.join(output_directory, out_filename)

# Write the extracted and rounded numbers to a new text file with headers
with open(output_filename, 'w') as output_file:
    output_file.write('Val Loss,Class Accuracy,Attr Accuracy\n')  # Writing headers
    for i in range(len(val_losses)):  # Assuming all lists are of the same length
        output_file.write(f"{val_losses[i]},{class_acc[i]},{attr_acc[i]}\n")
    output_file.write('\n\nCOMMS f1-score,PRONAV f1-score\n')  # Writing new headers
    for i in range(len(comms_f1_scores)):  # Assuming all lists are of the same length
        output_file.write(f"{comms_f1_scores[i]},{pronav_f1_scores[i]}\n")

print(f"Extraction complete. Check '{output_filename}' for the results.")

