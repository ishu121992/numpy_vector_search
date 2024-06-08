<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CLIP Embedding Search with Numpy and Pandas</title>
</head>
<body>
    <h1>CLIP Embedding Search with Numpy and Pandas</h1>
    <p>This repository demonstrates the power of the <strong>NumPy</strong> library and its <code>vstack</code> method, providing a robust alternative to vector databases when working with embeddings in a <code>Pandas</code> DataFrame.</p>

    <h2>Overview</h2>
    <p>This notebook uses the CLIP model to generate text embeddings and search for similar images within a dataset based on cosine similarity. By leveraging <strong>NumPy</strong> and <code>vstack</code>, the embeddings are efficiently managed and searched.</p>

    <h2>Setup</h2>
    <p>Before running the code, ensure you have the necessary dependencies installed:</p>
    <pre><code>pip install torch pandas numpy scikit-learn matplotlib pillow</code></pre>

    <h2>Configuration</h2>
    <p>The configuration class holds paths to the image directory, dataset CSV file, and the saved model. Modify the paths as per your setup.</p>
    <pre><code class="language-python">
class Config:
    def __init__(self):
        self.image_path = Path('../Original/')
        self.data_csv = '../Tabdata/dataset_train_refined.csv'
        self.model_path = '../models/clip_refined_84k.pth'  # Provide the path to your trained model

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()
    </code></pre>

    <h2>Load Model and Data</h2>
    <p>Load the pre-trained CLIP model and the image-text embedding DataFrame.</p>
    <pre><code class="language-python">
# Load the saved model state_dict
model_state_dict = torch.load(config.model_path)

# Create an instance of the CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the saved model state_dict into the model
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

embed_clip = pd.read_pickle('../Tabdata/embed_img_text_84k.pkl.gz')

# Read the dataset
df = pd.read_csv(config.data_csv)
image_file_names = [config.image_path / i for i in df['figure_file']]
labels = df['refined_object_title'].values.tolist()
ids = df['id'].values.tolist()
    </code></pre>

    <h2>Generate Text Embeddings</h2>
    <p>The function <code>generate_text_embeddings</code> generates text embeddings for given labels using the CLIP model.</p>
    <pre><code class="language-python">
def generate_text_embeddings(labels, batch_size=32):
    tok_text = clip.tokenize(labels).to(device)
    num_batches = (len(labels) + batch_size - 1) // batch_size
    text_features = torch.empty((len(labels), model.text_projection.shape[-1])).to(device)

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(labels))
            batch_tok_text = tok_text[start_idx:end_idx]
            batch_text_features = model.encode_text(batch_tok_text)
            text_features[start_idx:end_idx] = batch_text_features

    return text_features

prompt_embedding = generate_text_embeddings(['sports duffle bag']).cpu().numpy()
    </code></pre>

    <h2>Search Using NumPy and vstack</h2>
    <p>The <code>vstack</code> method from <strong>NumPy</strong> is utilized to create a single array of image embeddings. This array is then used to calculate cosine similarity with the text embedding.</p>
    <pre><code class="language-python">
# Convert the 'image_embeddings' column to a numpy array
img_embeddings_array = np.vstack(embed_clip['image_embeddings'].to_numpy())

# Calculate cosine similarity between the new embedding and all embeddings in the DataFrame
def vsearch(prompt_embedding, img_embeddings_array=img_embeddings_array):
    similarity_scores = cosine_similarity(prompt_embedding.reshape(1, -1), img_embeddings_array)
    df['cosine_similarity'] = similarity_scores[0]
    df_sorted = df.sort_values(by='cosine_similarity', ascending=False)
    return df_sorted

df_sorted = vsearch(prompt_embedding)
    </code></pre>

    <h2>Results</h2>
    <p>The sorted DataFrame contains the images ranked by their similarity to the text prompt. This method demonstrates how effectively <strong>NumPy</strong> and <code>vstack</code> can handle and search through embeddings without the need for a specialized vector database.</p>

    <h2>Conclusion</h2>
    <p>Using <strong>NumPy</strong> in combination with <code>Pandas</code> provides a powerful and efficient way to manage and search embeddings, offering a versatile alternative to dedicated vector databases.</p>
</body>
</html>
