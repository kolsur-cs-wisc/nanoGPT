import subprocess
import re
import matplotlib.pyplot as plt

training_losses = []
validation_losses = []

command = f"python train.py config/finetune_got.py"
proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

train_loss_pattern = re.compile(r".*train loss (\d+\.\d+).*")
val_loss_pattern = re.compile(r".*val loss (\d+\.\d+).*")

def extract_loss(line, pattern):
    match = pattern.search(line)
    if match:
        return float(match.group(1))
    return None

for line in proc.stdout:
    training_loss = extract_loss(line, train_loss_pattern)
    validation_loss = extract_loss(line, val_loss_pattern)
    if training_loss is not None:
        training_losses.append(training_loss)
    if validation_loss is not None:
        validation_losses.append(validation_loss)

iterations = [x for x in range(0, 151, 5)]
proc.stdout.close()
print(len(training_losses))
print(len(validation_losses))

plt.figure(figsize=(10, 6))
plt.plot(iterations, training_losses, label='Training Loss', color='blue')
plt.plot(iterations, validation_losses, label='Validation Loss', color='red')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid()

plot_path = 'finetuning.png'
plt.savefig(plot_path)
plt.show()