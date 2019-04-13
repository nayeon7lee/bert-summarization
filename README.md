## Implementation of 'Pretraining-Based Natural Language Generation for Text Summarization'

### Versions
* python 2.7
* PyTorch: 1.0.1.post2

### Preparing package/dataset
0. Run: `pip install -r requirements.txt` to install required packages
1. Download chunk CNN/DailyMail data from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail
2. Run: `python news_data_reader.py` to create pickle file that will be used in my data-loader

### Running the model
`python main.py --cuda --batch_size=2 --noam`
