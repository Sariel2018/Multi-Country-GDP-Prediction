# Meta-Prompting

[![arXiv](https://img.shields.io/badge/arXiv-2409.02551v1-b31b1b.svg)](https://arxiv.org/abs/2409.02551v1) **Deep Learning for Multi-Country GDP Prediction: A Study of Model Performance and Data Impact**


**Deep Learning for Multi-Country GDP Prediction** In this paper we give a comprehensive study on the GDP growth forecasting in the multi-country scenario by deep learning algorithms.

## Table of Contents

[**Abstract**](#abstract) | [**Dataset**](#datasets-and-process) | [**Runing and Evaluation**](#running-experiments-and-evaluation) | [**Citation**](#citation-guidelines) | [**Thanks**](#acknowledgements)


## Abstract
GDP is a vital measure of a country's economic health, reflecting the total value of goods and services produced. Forecasting GDP growth is essential for economic planning, as it helps governments, businesses, and investors anticipate trends, make informed decisions, and promote stability and growth. While most previous works focus on the prediction of the GDP growth rate for a single country or by machine learning methods, in this paper we give a comprehensive study on the GDP growth forecasting in the multi-country scenario by deep learning algorithms. For the prediction of the GDP growth where only GDP growth values are used, linear regression is generally better than deep learning algorithms. However, for the regression and the prediction of the GDP growth with selected economic indicators, deep learning algorithms could be superior to linear regression. We also investigate the influence of the novel data -- the light intensity data on the prediction of the GDP growth, and numerical experiments indicate that they do not necessarily improve the prediction performance.

## Datasets and Process

### GDP Datasets

We manually selected 13 annual economic indicators from the WEO database of the IMF and World Bank. These indicators are: `Rural population growth (annual $\%$)', `General government final consumption expenditure (annual $\%$ growth)', `Consumer price index (2010 = 100)', `Exports of goods and services (annual $\%$ growth)', `Urban population growth (annual $\%$)', `GDP growth (annual $\%$)', `Population growth (annual $\%$)', `Inflation, GDP deflator (annual $\%$)', `Imports of goods and services (annual $\%$ growth)', `Final consumption expenditure (annual $\%$ growth)', `Unemployment, total ($\%$ of total labor force) (national estimate)', `Inflation, consumer prices (annual $\%$)', `Gross fixed capital formation (annual $\%$ growth)' and `Households and NPISHs Final consumption expenditure (annual $\%$ growth)'.

Data access: 

https://data.worldbank.org/

https://www.imf.org/en/Publications/SPROLLs/world-economic-outlook-databases#sort=%40imfdate%20descending


For the quarterly data, we selected 20 economic indicators, including `Export Value', `Industrial Added Value', `Stock Market Capitalization', `Balance of Payments - Financial Account Balance', `Balance of Payments - Current Account Balance', `Balance of Payments - Current Account Credit', `Balance of Payments - Current Account Debit', `Balance of Payments - Capital Account Balance', `Balance of Payments - Capital Account Credit', `Balance of Payments - Capital Account Debit', `Overall Balance of Payments', `International Investment Position - Assets', `International Investment Position - Liabilities', `Net International Investment Position', `Import Value', `Nominal Effective Exchange Rate', `Retail Sales', `CPI (Consumer Price Index)', `Unemployment Rate' and `Central Bank Policy Rate'. The data was sourced from the financial institution WIND, with the original sources traceable to the World Bank, IMF, and national statistical offices.

### Lights data

The nighttime lights data is derived from monthly night-time remote sensing images captured by the Visible Infrared Imaging Radiometer Suite (VIIRS) onboard the NPP satellite. This data has a spatial resolution of 0.004 degrees and covers the period from the launch of the NPP satellite in 2012 to the present. The data has undergone stray light correction, which allows for an accurate representation of temporal changes in local lighting brightness. Nighttime lights data encompasses comprehensive information such as population, transportation, and economic development levels, making it suitable for various development economics research. The raw data is stored in the widely used GeoTIFF format for geographic information and is converted to brightness values using the Rasterio library of Python.

Data access: https://eogdata.mines.edu/products/vnl/

### Process data
All process data instructions used in our experiments are located in the /process_origin_data directory.


<!-- ## Results

All Results are stored in the `/outputs` directory.
-->


## Running and Evaluation

1. Install requirements. pip install -r requirements.txt

2. Prepare data.

3. Use different model to run results. All the scripts of linear, mlp, lstm and transformers are in the directory ./scripts. The scripts of timesfm, PatchTSt and Time-LLM model are in the directory ./timesfm&PatchTST&Time-LLM/{different-model}. For example, if you want to get the quarterly gdp results of lstm models, just run the following command, and you can open ./checkpoints_lstm/. to see the results once all is done:

```python
python run_lstm_q.py
```

**Note:** Please make sure the dataset path is right. 
**Note:** In Time-LLM, we use describe_add as our prompt.


## Citation Guidelines

We encourage citing our paper if our data or findings are used in your research. Please also acknowledge the original authors and models used and referenced in our work.

```bibtex
@article{xie2024deep,
  title={Deep Learning for Multi-Country GDP Prediction: A Study of Model Performance and Data Impact},
  author={Xie, Huaqing and Xu, Xingcheng and Yan, Fangjia and Qian, Xun and Yang, Yanqing},
  journal={arXiv preprint arXiv:2409.02551},
  year={2024}
}
```

## Acknowledgements

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/yuqinie98/PatchTST

https://github.com/KimMeen/Time-LLM

https://github.com/google-research/timesfm
