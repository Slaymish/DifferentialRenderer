import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from torch.utils.data import Dataset, DataLoader

# 1. Define the vocabulary of mathematical symbols
symbols = [
    # Logical operators
    '¬',  # NOT
    '∧',  # AND
    '∨',  # OR
    '→',  # IMPLIES
    '↔',  # IFF (if and only if)

    # Quantifiers
    '∀',  # FOR ALL
    '∃',  # EXISTS

    # Parentheses
    '(', ')',

    # Propositional variables
    'P', 'Q', 'R', 'S', 'T',

    # Constants
    '⊤',  # TRUE
    '⊥',  # FALSE
]

# Add special tokens for start and end of sequence
special_tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
symbols.extend(special_tokens)

# Assign unique integer IDs to each symbol
symbol_to_id = {symbol: idx for idx, symbol in enumerate(symbols)}
id_to_symbol = {idx: symbol for symbol, idx in symbol_to_id.items()}

# 2. Encoding and Decoding Functions
def encode_symbols(symbol_sequence):
    """
    Encode a sequence of symbols into a sequence of integer IDs.
    """
    return [symbol_to_id.get(symbol, symbol_to_id['<UNK>']) for symbol in symbol_sequence]

def decode_ids(id_sequence):
    """
    Decode a sequence of integer IDs back into a sequence of symbols.
    """
    return [id_to_symbol.get(id_, '<UNK>') for id_ in id_sequence]

# 3. Define Logical Inference Rules
def modus_ponens(premise1, premise2):
    """
    Apply Modus Ponens inference rule.
    Premise1: P
    Premise2: P → Q
    Conclusion: Q
    """
    if len(premise1) == 1 and len(premise2) == 3:
        if premise1[0] == premise2[0] and premise2[1] == '→':
            conclusion = [premise2[2]]
            return conclusion
    return None

def modus_tollens(premise1, premise2):
    """
    Apply Modus Tollens inference rule.
    Premise1: ¬Q
    Premise2: P → Q
    Conclusion: ¬P
    """
    if len(premise1) == 2 and len(premise2) == 3:
        if premise1[0] == '¬' and premise2[1] == '→' and premise1[1] == premise2[2]:
            conclusion = ['¬', premise2[0]]
            return conclusion
    return None

def hypothetical_syllogism(premise1, premise2):
    """
    Apply Hypothetical Syllogism inference rule.
    Premise1: P → Q
    Premise2: Q → R
    Conclusion: P → R
    """
    if len(premise1) == 3 and len(premise2) == 3:
        if premise1[1] == '→' and premise2[1] == '→' and premise1[2] == premise2[0]:
            conclusion = [premise1[0], '→', premise2[2]]
            return conclusion
    return None

def disjunctive_syllogism(premise1, premise2):
    """
    Apply Disjunctive Syllogism inference rule.
    Premise1: P ∨ Q
    Premise2: ¬P
    Conclusion: Q
    """
    if len(premise1) == 3 and len(premise2) == 2:
        if premise1[1] == '∨' and premise2[0] == '¬' and premise2[1] == premise1[0]:
            conclusion = [premise1[2]]
            return conclusion
    return None

def addition(premise1):
    """
    Apply Addition inference rule.
    Premise1: P
    Conclusion: P ∨ Q
    """
    variables = ['P', 'Q', 'R', 'S', 'T']
    variables.remove(premise1[0])  # Ensure Q is different from P
    Q = random.choice(variables)
    conclusion = [premise1[0], '∨', Q]
    return conclusion

def simplification(premise1):
    """
    Apply Simplification inference rule.
    Premise1: P ∧ Q
    Conclusion: P
    """
    if len(premise1) == 3 and premise1[1] == '∧':
        conclusion = [premise1[0]]
        return conclusion
    return None

def conjunction(premise1, premise2):
    """
    Apply Conjunction inference rule.
    Premise1: P
    Premise2: Q
    Conclusion: P ∧ Q
    """
    if len(premise1) == 1 and len(premise2) == 1:
        conclusion = [premise1[0], '∧', premise2[0]]
        return conclusion
    return None

def generate_valid_proof():
    """
    Generate a valid proof using logical inference rules.
    """
    # Randomly choose an inference rule
    rules = [modus_ponens, modus_tollens, hypothetical_syllogism, disjunctive_syllogism, addition, simplification, conjunction]
    rule = random.choice(rules)

    variables = ['P', 'Q', 'R', 'S', 'T']
    random.shuffle(variables)

    if rule == modus_ponens:
        P, Q = variables[:2]
        step1 = [P]
        step2 = [P, '→', Q]
        step3 = rule(step1, step2)
        if step3 is None:
            return None
        proof = [step1, step2, step3]
    elif rule == modus_tollens:
        P, Q = variables[:2]
        step1 = ['¬', Q]
        step2 = [P, '→', Q]
        step3 = rule(step1, step2)
        if step3 is None:
            return None
        proof = [step1, step2, step3]
    elif rule == hypothetical_syllogism:
        P, Q, R = variables[:3]
        step1 = [P, '→', Q]
        step2 = [Q, '→', R]
        step3 = rule(step1, step2)
        if step3 is None:
            return None
        proof = [step1, step2, step3]
    elif rule == disjunctive_syllogism:
        P, Q = variables[:2]
        step1 = [P, '∨', Q]
        step2 = ['¬', P]
        step3 = rule(step1, step2)
        if step3 is None:
            return None
        proof = [step1, step2, step3]
    elif rule == addition:
        P = variables[0]
        step1 = [P]
        step2 = rule(step1)
        proof = [step1, step2]
    elif rule == simplification:
        P, Q = variables[:2]
        step1 = [P, '∧', Q]
        step2 = rule(step1)
        if step2 is None:
            return None
        proof = [step1, step2]
    elif rule == conjunction:
        P, Q = variables[:2]
        step1 = [P]
        step2 = [Q]
        step3 = rule(step1, step2)
        if step3 is None:
            return None
        proof = [step1, step2, step3]
    else:
        return None  # Unknown rule

    return proof

def is_valid_step(previous_steps, current_step):
    """
    Verify if the current step is valid based on previous steps.
    """
    if len(previous_steps) < 1:
        return True  # Can't apply inference rules yet

    for rule in [modus_ponens, modus_tollens, hypothetical_syllogism, disjunctive_syllogism, addition, simplification, conjunction]:
        if rule in [addition, simplification]:
            expected_conclusion = rule(previous_steps[-1])
            if expected_conclusion and current_step == expected_conclusion:
                return True
        else:
            if len(previous_steps) >= 2:
                last_two_steps = previous_steps[-2:]
                expected_conclusion = rule(*last_two_steps)
                if expected_conclusion and current_step == expected_conclusion:
                    return True
    return False

# 4. Generate Mathematically Valid Synthetic Dataset
def generate_synthetic_proofs(num_proofs=2000):
    """
    Generate a dataset of valid proofs.
    """
    proofs = []
    while len(proofs) < num_proofs:
        proof = generate_valid_proof()
        if proof:
            # Ensure none of the steps are None
            if all(step is not None for step in proof):
                proofs.append(proof)
    return proofs

# 5. Prepare Data for Training
def prepare_training_data(proofs):
    """
    Prepare input and target sequences for training.
    """
    inputs = []
    targets = []

    for proof in proofs:
        for i in range(len(proof) - 1):
            # Input: All steps up to the current one
            input_steps = proof[:i+1]
            input_sequence = ['<SOS>'] + [symbol for step in input_steps for symbol in step] + ['<EOS>']
            input_encoded = encode_symbols(input_sequence)

            # Target: The next step, with <SOS> and <EOS>
            if proof[i+1] is None:
                continue  # Skip if the next step is None
            target_step = ['<SOS>'] + proof[i+1] + ['<EOS>']
            target_encoded = encode_symbols(target_step)

            inputs.append(input_encoded)
            targets.append(target_encoded)

    return inputs, targets

# Custom Dataset
class ProofDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.max_input_length = max(len(seq) for seq in inputs)
        self.max_target_length = max(len(seq) for seq in targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        target_seq = self.targets[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

    def collate_fn(self, batch):
        batch_inputs, batch_targets = zip(*batch)
        batch_inputs_padded = nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value=symbol_to_id['<PAD>'])
        batch_targets_padded = nn.utils.rnn.pad_sequence(batch_targets, batch_first=True, padding_value=symbol_to_id['<PAD>'])
        return batch_inputs_padded, batch_targets_padded

# 6. Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dim_feedforward, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=symbol_to_id['<PAD>'])
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.embedding_dim = embedding_dim

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt) * math.sqrt(self.embedding_dim)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(
            src.transpose(0, 1),
            tgt.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = self.fc_out(output)
        return output.transpose(0, 1)

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(1, max_len, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 0:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 7. Training Loop with Early Stopping and Learning Rate Scheduler
def train_model(model, train_loader, epochs=50, patience=5):
    criterion = nn.CrossEntropyLoss(ignore_index=symbol_to_id['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_inputs_padded, batch_targets_padded in train_loader:
            batch_inputs_padded = batch_inputs_padded.to(model.device)
            batch_targets_padded = batch_targets_padded.to(model.device)

            # Create masks
            src_key_padding_mask = (batch_inputs_padded == symbol_to_id['<PAD>'])
            tgt_key_padding_mask = (batch_targets_padded == symbol_to_id['<PAD>'])
            memory_key_padding_mask = src_key_padding_mask

            # Forward pass
            outputs = model(
                batch_inputs_padded,
                batch_targets_padded[:, :-1],
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
                memory_key_padding_mask=memory_key_padding_mask,
            )

            # Shift targets to align with outputs
            outputs = outputs.reshape(-1, outputs.shape[-1])
            batch_targets_flat = batch_targets_padded[:, 1:].reshape(-1)

            loss = criterion(outputs, batch_targets_flat)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# 8. Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_inputs_padded, batch_targets_padded in test_loader:
            batch_inputs_padded = batch_inputs_padded.to(model.device)
            batch_targets_padded = batch_targets_padded.to(model.device)

            for inp, tgt in zip(batch_inputs_padded, batch_targets_padded):
                predicted_symbols = predict_next_step(model, inp)
                target_symbols = decode_ids(tgt.tolist())
                # Exclude special tokens
                target_symbols = [sym for sym in target_symbols if sym not in special_tokens]
                predicted_symbols = [sym for sym in predicted_symbols if sym not in special_tokens]

                if predicted_symbols == target_symbols:
                    correct += 1
                else:
                    # Optional: Print incorrect predictions for analysis
                    print(f"Input: {decode_ids(inp.tolist())}")
                    print(f"Target: {target_symbols}")
                    print(f"Predicted: {predicted_symbols}\n")
                total += 1

    accuracy = correct / total * 100
    print(f"Evaluation Accuracy: {accuracy:.2f}%")


# 9. Prediction Function with Beam Search
def predict_next_step(model, input_sequence, max_length=10, beam_width=5):
    """
    Predict the next step using beam search.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = input_sequence.unsqueeze(0)  # Add batch dimension
        src_key_padding_mask = (input_tensor == symbol_to_id['<PAD>']).to(model.device)
        input_tensor = input_tensor.to(model.device)

        # Initialize the beam
        beams = [(['<SOS>'], 0.0)]

        for _ in range(max_length):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == '<EOS>':
                    new_beams.append((seq, score))
                    continue

                tgt_sequence = encode_symbols(seq)
                tgt_tensor = torch.tensor(tgt_sequence, dtype=torch.long).unsqueeze(0).to(model.device)
                tgt_key_padding_mask = (tgt_tensor == symbol_to_id['<PAD>']).to(model.device)

                output = model(
                    input_tensor,
                    tgt_tensor,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                next_token_logits = output[0, -1, :]
                probs = nn.functional.log_softmax(next_token_logits, dim=-1)
                topk = torch.topk(probs, beam_width)

                for idx in range(beam_width):
                    next_token = id_to_symbol[topk.indices[idx].item()]
                    next_score = score + topk.values[idx].item()
                    new_seq = seq + [next_token]
                    new_beams.append((new_seq, next_score))

            # Keep the top k beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Check if all beams have ended
            if all(seq[-1] == '<EOS>' for seq, _ in beams):
                break

        # Select the best beam
        best_seq = beams[0][0]
        predicted_symbols = [sym for sym in best_seq if sym not in special_tokens]
        return predicted_symbols

# Main execution
if __name__ == '__main__':
    # Generate synthetic proofs
    synthetic_proofs = generate_synthetic_proofs(num_proofs=20000)

    random.shuffle(synthetic_proofs)

    # Split the proofs into training and testing sets
    split_index = int(len(synthetic_proofs) * 0.8)
    train_proofs = synthetic_proofs[:split_index]
    test_proofs = synthetic_proofs[split_index:]

    # Prepare training data
    train_inputs, train_targets = prepare_training_data(train_proofs)

    # Prepare testing data
    test_inputs, test_targets = prepare_training_data(test_proofs)

    # Create datasets and dataloaders
    train_dataset = ProofDataset(train_inputs, train_targets)
    test_dataset = ProofDataset(test_inputs, test_targets)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Define model parameters
    vocab_size = len(symbols)
    embedding_dim = 256
    num_heads = 8
    num_layers = 6
    dim_feedforward = 1024

    # Initialize the model
    model = TransformerModel(
        vocab_size,
        embedding_dim,
        num_heads,
        num_layers,
        dim_feedforward,
    )

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model.to(device)

    # Train the model
    train_model(model, train_loader, epochs=100, patience=5)

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model
    evaluate_model(model, test_loader)

    # Test the model with an example
    test_proof = [
        ['P'],
        ['P', '→', 'Q'],
    ]
    # Flatten and encode the test proof
    test_input_sequence = ['<SOS>'] + [symbol for step in test_proof for symbol in step] + ['<EOS>']
    test_input_encoded = encode_symbols(test_input_sequence)
    test_input_encoded = torch.tensor(test_input_encoded, dtype=torch.long).to(device)

    # Predict the next step
    predicted_symbols = predict_next_step(model, test_input_encoded)

    # Output the result
    print("\nTest Proof Input Steps:", test_proof)
    print("Predicted Next Step:", predicted_symbols)

    # Verify the prediction
    is_valid = is_valid_step(test_proof, predicted_symbols)
    print("Is the predicted step valid?", is_valid)
