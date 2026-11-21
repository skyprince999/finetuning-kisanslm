# EDA report: `kcc_data_apple.jsonl`


## 1. Dataset overview

- Rows: **308,448**
- Columns: **22** -> ['StateName', 'DistrictName', 'BlockName', 'Season', 'Sector', 'Category', 'Crop', 'QueryType', 'QueryText', 'KccAns', 'CreatedOn', 'year', 'month', 'date', 'hour', 'weekday', 'QueryText_chars', 'QueryText_words', 'QueryText_script', 'KccAns_chars', 'KccAns_words', 'KccAns_script']
- Malformed JSON lines skipped: 0
- Memory: 181.9 MB

### 2. Missing / empty / 'NA' values

| column           |   null |   empty |   literal_NA |   unique |
|:-----------------|-------:|--------:|-------------:|---------:|
| StateName        |      0 |       0 |          340 |       36 |
| DistrictName     |      0 |       0 |          554 |      622 |
| BlockName        |      0 |       0 |        28973 |     3156 |
| Season           |      0 |       0 |       171041 |        4 |
| Sector           |      0 |       0 |            0 |        3 |
| Category         |      0 |       0 |        10964 |        6 |
| Crop             |      0 |       0 |            0 |        2 |
| QueryType        |      0 |       0 |        17368 |       73 |
| QueryText        |      0 |       3 |            0 |    77373 |
| KccAns           |      0 |     141 |         2904 |   118530 |
| CreatedOn        |      0 |       0 |            0 |   303572 |
| year             |      0 |       0 |            0 |       17 |
| month            |      0 |       0 |            0 |       12 |
| date             |      0 |       0 |            0 |     5784 |
| hour             |      0 |       0 |            0 |       24 |
| weekday          |      0 |       0 |            0 |        7 |
| QueryText_chars  |      0 |       0 |            0 |      342 |
| QueryText_words  |      0 |       0 |            0 |       73 |
| QueryText_script |      0 |       0 |            0 |        3 |
| KccAns_chars     |      0 |       0 |            0 |     1832 |
| KccAns_words     |      0 |       0 |            0 |      577 |
| KccAns_script    |      0 |       0 |            0 |        5 |


## 3. Categorical distributions


### StateName (unique=36)

| StateName         |   count |   pct |
|:------------------|--------:|------:|
| JAMMU AND KASHMIR |  162395 | 52.65 |
| HIMACHAL PRADESH  |  131526 | 42.64 |
| MAHARASHTRA       |    5574 |  1.81 |
| UTTARAKHAND       |    1473 |  0.48 |
| GUJARAT           |     978 |  0.32 |
| UTTAR PRADESH     |     843 |  0.27 |
| RAJASTHAN         |     751 |  0.24 |
| WEST BENGAL       |     722 |  0.23 |
| ODISHA            |     503 |  0.16 |
| BIHAR             |     501 |  0.16 |
| KARNATAKA         |     473 |  0.15 |
| TAMILNADU         |     429 |  0.14 |
| MADHYA PRADESH    |     387 |  0.13 |
| NA                |     338 |  0.11 |
| ANDHRA PRADESH    |     301 |  0.1  |


### DistrictName (unique=622)

| DistrictName   |   count |   pct |
|:---------------|--------:|------:|
| SHIMLA         |   60322 | 19.56 |
| BADGAM         |   33485 | 10.86 |
| PULWAMA        |   29620 |  9.6  |
| MANDI          |   27745 |  9    |
| KULLU          |   26022 |  8.44 |
| BARAMULLA      |   24471 |  7.93 |
| ANANTNAG       |   21191 |  6.87 |
| KULGAM         |   18213 |  5.9  |
| SHUPIYAN       |   17944 |  5.82 |
| KINNAUR        |    4791 |  1.55 |
| GANDERBAL      |    4770 |  1.55 |
| SIRMAUR        |    4142 |  1.34 |
| CHAMBA         |    3849 |  1.25 |
| KUPWARA        |    3583 |  1.16 |
| BANDIPORE      |    3307 |  1.07 |


### BlockName (unique=3156)

| BlockName      |   count |   pct |
|:---------------|--------:|------:|
| 0              |   28631 |  9.28 |
| PULWAMA        |   17272 |  5.6  |
| SHOPIAN        |   16290 |  5.28 |
| CHAUPAL        |   12621 |  4.09 |
| KARSOG         |   12110 |  3.93 |
| THEOG          |   10658 |  3.46 |
| KULGAM         |    9485 |  3.08 |
| ANNI           |    9280 |  3.01 |
| SERAJ          |    9019 |  2.92 |
| ROHRU          |    8753 |  2.84 |
| JUBBAL KOTKHAI |    7298 |  2.37 |
| BEERWAH        |    7281 |  2.36 |
| NARKANDA       |    7032 |  2.28 |
| ACHABAL        |    6914 |  2.24 |
| NIRMAND        |    6522 |  2.11 |


### Season (unique=4)

| Season   |   count |   pct |
|:---------|--------:|------:|
| NA       |  171041 | 55.45 |
| KHARIF   |   52398 | 16.99 |
| JAYAD    |   47741 | 15.48 |
| RABI     |   37268 | 12.08 |


### Sector (unique=3)

| Sector       |   count |   pct |
|:-------------|--------:|------:|
| HORTICULTURE |  301978 |  97.9 |
| AGRICULTURE  |    6469 |   2.1 |
| FISHERIES    |       1 |   0   |


### Category (unique=6)

| Category               |   count |   pct |
|:-----------------------|--------:|------:|
| Fruits                 |  297351 | 96.4  |
| 0                      |   10964 |  3.55 |
| Others                 |     123 |  0.04 |
| Vegetables             |       8 |  0    |
| Sugar and Starch Crops |       1 |  0    |
| Oilseeds               |       1 |  0    |


### Crop (unique=2)

| Crop          |   count |   pct |
|:--------------|--------:|------:|
| Apple         |  299548 | 97.11 |
| Custard Apple |    8900 |  2.89 |


### QueryType (unique=73)

| QueryType                       |   count |   pct |
|:--------------------------------|--------:|------:|
| Plant Protection                |  168402 | 54.6  |
| Cultural Practices              |   36145 | 11.72 |
| Weather                         |   28376 |  9.2  |
| 0                               |   17368 |  5.63 |
| Fertilizer Use and Availability |   13279 |  4.31 |
| 75                              |   10378 |  3.36 |
| Nutrient Management             |   10241 |  3.32 |
| 76                              |    4028 |  1.31 |
| Seeds and Planting Material     |    2667 |  0.86 |
| Market Information              |    1564 |  0.51 |
| 102                             |    1443 |  0.47 |
| 85                              |    1374 |  0.45 |
| Agriculture Mechanization       |    1201 |  0.39 |
| Government Schemes              |    1147 |  0.37 |
| 2                               |    1113 |  0.36 |


### Top districts per state

|                                                 |   count |
|:------------------------------------------------|--------:|
| ('0', '0')                                      |       2 |
| ('A AND N ISLANDS', 'NORTH AND MIDDLE ANDAMAN') |       1 |
| ('ANDHRA PRADESH', 'GUNTUR')                    |      39 |
| ('ANDHRA PRADESH', 'NELLORE')                   |      38 |
| ('ANDHRA PRADESH', 'ANANTPUR')                  |      32 |
| ('ANDHRA PRADESH', 'CHITTOOR')                  |      32 |
| ('ANDHRA PRADESH', 'EAST GODAVARI')             |      31 |
| ('ARUNACHAL PRADESH', '9999')                   |       1 |
| ('ARUNACHAL PRADESH', 'TAWANG')                 |       1 |
| ('ARUNACHAL PRADESH', 'UPPER SIANG')            |       1 |
| ('ASSAM', 'NAGAON')                             |       6 |
| ('ASSAM', 'DIBRUGARH')                          |       5 |
| ('ASSAM', 'KARIMGANJ')                          |       5 |
| ('ASSAM', 'CACHAR')                             |       4 |
| ('ASSAM', 'TINSUKIA')                           |       4 |
| ('BIHAR', 'SAMASTIPUR')                         |      49 |
| ('BIHAR', 'PURBA CHAMPARAN')                    |      32 |
| ('BIHAR', 'MUZAFFARPUR')                        |      29 |
| ('BIHAR', 'VAISHALI')                           |      29 |
| ('BIHAR', 'BHAGALPUR')                          |      25 |
| ('CHANDIGARH', 'CHANDIGARH')                    |       3 |
| ('CHHATTISGARH', 'JANJGIR-CHAMPA')              |      10 |
| ('CHHATTISGARH', 'RAIGARH')                     |       9 |
| ('CHHATTISGARH', 'MAHASAMUND')                  |       8 |
| ('CHHATTISGARH', 'SURGUJA')                     |       8 |
| ('CHHATTISGARH', 'Surajpur')                    |       7 |
| ('DAMAN AND DIU', 'DAMAN')                      |       1 |
| ('DAMAN AND DIU', 'DIU')                        |       1 |
| ('DELHI', 'New Delhi (Connaught Place)')        |      53 |
| ('DELHI', 'EAST DELHI (Preet Vihar)')           |      15 |
| ('DELHI', 'CENTRAL DELHI (Daryaganj)')          |      11 |
| ('DELHI', 'North Delhi (Narela)')               |       8 |
| ('DELHI', 'South Delhi (Saket)')                |       3 |
| ('GOA', 'GOA NORTH')                            |       1 |
| ('GUJARAT', 'JUNAGADH')                         |     193 |
| ('GUJARAT', 'RAJKOT')                           |     126 |
| ('GUJARAT', 'BHAVNAGAR')                        |     115 |
| ('GUJARAT', 'JAMNAGAR')                         |      83 |
| ('GUJARAT', 'AMRELI')                           |      55 |
| ('HARYANA', 'HISSAR')                           |      37 |
| ('HARYANA', 'BHIWANI')                          |      35 |
| ('HARYANA', 'SIRSA')                            |      25 |
| ('HARYANA', 'MAHENDRA GARH')                    |      19 |
| ('HARYANA', 'JIND')                             |      16 |
| ('HIMACHAL PRADESH', 'SHIMLA')                  |   60322 |
| ('HIMACHAL PRADESH', 'MANDI')                   |   27745 |
| ('HIMACHAL PRADESH', 'KULLU')                   |   26022 |
| ('HIMACHAL PRADESH', 'KINNAUR')                 |    4791 |
| ('HIMACHAL PRADESH', 'SIRMAUR')                 |    4142 |
| ('JAMMU AND KASHMIR', 'BADGAM')                 |   33485 |
| ('JAMMU AND KASHMIR', 'PULWAMA')                |   29620 |
| ('JAMMU AND KASHMIR', 'BARAMULLA')              |   24471 |
| ('JAMMU AND KASHMIR', 'ANANTNAG')               |   21191 |
| ('JAMMU AND KASHMIR', 'KULGAM')                 |   18213 |
| ('JHARKAND', 'PALAMU')                          |      16 |
| ('JHARKAND', 'RANCHI')                          |      13 |
| ('JHARKAND', 'HAZARIBAGH')                      |       9 |
| ('JHARKAND', 'KODARMA')                         |       7 |
| ('JHARKAND', 'GIRIDIH')                         |       6 |
| ('KARNATAKA', 'BIJAPUR')                        |      78 |


### QueryType mix by top-6 states (% of state's queries)

| QueryType                       |   JAMMU AND KASHMIR |   HIMACHAL PRADESH |   MAHARASHTRA |   UTTARAKHAND |   GUJARAT |   UTTAR PRADESH |
|:--------------------------------|--------------------:|-------------------:|--------------:|--------------:|----------:|----------------:|
| 0                               |                 7.5 |                3.4 |           5.7 |           9.7 |       5.1 |             3.4 |
| 102                             |                 0.5 |                0.4 |           0.5 |           0.8 |       1.8 |             0.6 |
| 75                              |                 2.9 |                4.2 |           0.2 |           7.7 |       1.2 |             0.8 |
| 76                              |                 0.8 |                1.9 |           0.3 |           4.5 |       0.8 |             0.6 |
| 85                              |                 0.2 |                0.8 |           0.1 |           0.1 |       0.1 |             0   |
| Cultural Practices              |                 9.7 |               14.1 |           7.2 |          11.5 |      26.6 |            14.8 |
| Fertilizer Use and Availability |                 3.1 |                5.4 |          13.8 |           4.4 |       7.6 |             2.3 |
| Market Information              |                 0.7 |                0.1 |           3.6 |           0.7 |       0.8 |             2.1 |
| Nutrient Management             |                 2.4 |                3.9 |           4.8 |           1.9 |       1.9 |             7.4 |
| Plant Protection                |                56.4 |               54.8 |          29.1 |          30.1 |      30.7 |            39.6 |
| Seeds and Planting Material     |                 0.1 |                0.8 |          18.8 |           1.9 |       4.6 |             3.1 |
| Weather                         |                12.3 |                6.1 |           0.6 |           2.9 |       1.3 |             3.9 |

- QueryType values that are bare numeric codes (e.g. '0', '75'): 40,234 rows (13.04%)

## 4. Temporal analysis

- Unparseable timestamps: 0
- Date range: 2009-01-04 11:19:59  ->  2025-07-23 21:09:44
- Distinct days with calls: 5,784

### Queries per year

|   CreatedOn |   count |
|------------:|--------:|
|        2009 |    3076 |
|        2010 |    6187 |
|        2011 |    4368 |
|        2012 |    6997 |
|        2013 |   30761 |
|        2014 |   42462 |
|        2015 |   25703 |
|        2016 |   18955 |
|        2017 |   19889 |
|        2018 |   16742 |
|        2019 |   30961 |
|        2020 |   11814 |
|        2021 |   19562 |
|        2022 |   16346 |
|        2023 |   20441 |
|        2024 |   17881 |
|        2025 |   16303 |


### Seasonality – queries by calendar month (all years)

|   CreatedOn |   count |
|------------:|--------:|
|           1 |   16893 |
|           2 |   17653 |
|           3 |   39572 |
|           4 |   55358 |
|           5 |   46778 |
|           6 |   40168 |
|           7 |   32111 |
|           8 |   17097 |
|           9 |    9526 |
|          10 |    7984 |
|          11 |   10987 |
|          12 |   14321 |


### Hour of day / weekday

|   hour |   count |
|-------:|--------:|
|      0 |     288 |
|      1 |     115 |
|      2 |      77 |
|      3 |     121 |
|      4 |     118 |
|      5 |     125 |
|      6 |    4711 |
|      7 |   12226 |
|      8 |   20743 |
|      9 |   26737 |
|     10 |   32441 |
|     11 |   30817 |
|     12 |   26465 |
|     13 |   21519 |
|     14 |   21765 |
|     15 |   22232 |
|     16 |   21896 |
|     17 |   19864 |
|     18 |   16847 |
|     19 |   12172 |
|     20 |   10071 |
|     21 |    6859 |
|     22 |     236 |
|     23 |       3 |

| weekday   |   count |
|:----------|--------:|
| Monday    |   44415 |
| Tuesday   |   44368 |
| Wednesday |   44670 |
| Thursday  |   43893 |
| Friday    |   44509 |
| Saturday  |   44912 |
| Sunday    |   41681 |

- Rows where year/month fields disagree with CreatedOn: 0

## 5. Text analysis (QueryText / KccAns)


### QueryText length statistics

|       |   QueryText_chars |   QueryText_words |
|:------|------------------:|------------------:|
| count |          308448   |          308448   |
| mean  |              45.9 |               7.6 |
| std   |              27.7 |               4.5 |
| min   |               0   |               0   |
| 5%    |              18   |               3   |
| 25%   |              31   |               5   |
| 50%   |              43   |               8   |
| 75%   |              57   |               9   |
| 95%   |              79   |              13   |
| 99%   |              99   |              16   |
| max   |            7316   |            1421   |


### QueryText script / language

| QueryText_script   |   count |   pct |
|:-------------------|--------:|------:|
| english(latin)     |  307811 | 99.79 |
| other/numeric      |     634 |  0.21 |
| empty              |       3 |  0    |


### KccAns length statistics

|       |   KccAns_chars |   KccAns_words |
|:------|---------------:|---------------:|
| count |       308448   |       308448   |
| mean  |          178.4 |           31.5 |
| std   |          260.7 |           44.5 |
| min   |            0   |            0   |
| 5%    |           30   |            5   |
| 25%   |           58   |           11   |
| 50%   |          112   |           21   |
| 75%   |          218   |           39   |
| 95%   |          519   |           88   |
| 99%   |         1066   |          163   |
| max   |        23255   |         3529   |


### KccAns script / language

| KccAns_script     |   count |   pct |
|:------------------|--------:|------:|
| english(latin)    |  257019 | 83.33 |
| hindi(devanagari) |   42588 | 13.81 |
| other/numeric     |    4365 |  1.42 |
| mixed             |    4335 |  1.41 |
| empty             |     141 |  0.05 |


### Answer quality flags

|                                       |   rows |   pct |
|:--------------------------------------|-------:|------:|
| answer empty                          |    141 |  0.05 |
| answer < 5 words                      |  13601 |  4.41 |
| answer > 300 words                    |    925 |  0.3  |
| answer mentions call/contact/transfer |   6117 |  1.98 |
| answer contains phone-like number     |   1462 |  0.47 |
| answer == query                       |   1241 |  0.4  |


### Shortest answers (sample)

|     | QueryText                                          | KccAns                         |
|----:|:---------------------------------------------------|:-------------------------------|
|  54 | Regarding less size of apple                       | You can spray multinutrient    |
| 234 | Incomplete call                                    | Incomplete call                |
| 258 | HOW TO CONTROL LEAF SPOTS IN APPLE TREES ?         | Bitertanol 25WP (50 g)         |
| 384 | SPRAY SCHEDULE IN APPLE?                           | CALCIUM + BORON .              |
| 455 | SPRAY SCHEDULE IN APPLE                            | PLANOFIX@1ML/4.5LITRE OF WATER |
| 621 | CONTACT NUMBER OF KVK BAJAURA ?                    | KINDLY CONTACT 01905 287318    |
| 644 | TEST CALL                                          | TEST CALL                      |
| 721 | HOW TO CONTROL PIN SIZE BLACK SPOT ON APPLE TREE ? | NA                             |
| 760 | PM KISAN STATUS                                    | १८००११५५२६ पं किसान स्टेटस     |
| 852 | spray schedule in apple                            | solubar@3g/1 litre of water    |


## 6. Frequent terms & queries


### Top 40 query terms (stopwords removed)

|             |   count |
|:------------|--------:|
| spray       |  109802 |
| control     |   48682 |
| stage       |   47447 |
| weather     |   39755 |
| fruit       |   36062 |
| dose        |   32861 |
| schedule    |   25927 |
| development |   18690 |
| fall        |   17929 |
| after       |   17712 |
| fertilizer  |   17436 |
| time        |   14203 |
| petal       |   13315 |
| size        |   13085 |
| pink        |   12806 |
| much        |   12508 |
| root        |   10738 |
| bud         |   10342 |
| report      |    9300 |
| walnut      |    8803 |
| leaf        |    8516 |
| rot         |    8505 |
| days        |    7787 |
| harvest     |    7625 |
| scab        |    7312 |
| pea         |    7135 |
| fungicide   |    6942 |
| forecast    |    6237 |
| canker      |    6151 |
| application |    5245 |
| aphid       |    4871 |
| custard     |    4822 |
| harvesting  |    4748 |
| tip         |    4612 |
| before      |    4592 |
| green       |    4344 |
| year        |    4206 |
| management  |    3845 |
| problem     |    3837 |
| recommended |    3687 |


### Top 25 query bigrams

|                    |   count |
|:-------------------|--------:|
| spray schedule     |   24971 |
| fruit development  |   18327 |
| petal fall         |   13120 |
| fertilizer dose    |   12840 |
| fall stage         |   10474 |
| pink bud           |   10201 |
| spray after        |    9418 |
| weather report     |    9123 |
| bud stage          |    9051 |
| control root       |    8647 |
| spray pink         |    8403 |
| spray petal        |    7703 |
| spray fruit        |    7527 |
| root rot           |    7034 |
| weather forecast   |    6152 |
| size stage         |    5879 |
| pea size           |    5763 |
| spray dose         |    5187 |
| after fruit        |    5006 |
| walnut size        |    5003 |
| control canker     |    4946 |
| control scab       |    4426 |
| dose fruit         |    4379 |
| green tip          |    4182 |
| development walnut |    4066 |


### Top 20 normalized queries

| QueryText                                                                                     |   count |
|:----------------------------------------------------------------------------------------------|--------:|
| farmer asked query on weather                                                                 |   15028 |
| information regarding weather report                                                          |    5002 |
| weather forecast                                                                              |    4486 |
| spray schedule in apple                                                                       |    3026 |
| information regarding spray at pink bud stage in apple                                        |    3004 |
| regarding spray schedule of apple                                                             |    2846 |
| information regarding fertilizer dose of apple plants                                         |    2360 |
| spray schedule in apple trees                                                                 |    2257 |
| regarding spray in apple tree                                                                 |    2257 |
| information regarding spray of petal fall stage in apple                                      |    2191 |
| information regarding spray after fruit development 20 days after walnut stage in apple plant |    2155 |
| weather report                                                                                |    2044 |
| spray schedule in apple orchard                                                               |    1753 |
| information regarding spray at petal fall stage in apple                                      |    1733 |
| how to control root rot in apple                                                              |    1502 |
| information regarding recommendations against hail damage in apple                            |    1470 |
| information regarding spray dose of fruit development walnut size in apple                    |    1405 |
| how much fertilizer dose of apple plants                                                      |    1364 |
| todays weather forecast                                                                       |    1209 |
| regarding spray schedule in apple                                                             |    1206 |


## 7. Duplicates & answer diversity

- Fully identical records: 652 (0.21%)
- Duplicate (QueryText, KccAns) pairs: 152,929 (49.58%)
- Duplicate normalized QueryText: 250,467 (81.20%)
- Unique normalized queries: 57,981
- Unique answers: 118,530

### Answer diversity for frequent queries (>=20 occurrences)

- Frequent queries: 1,591; median distinct answers each: 17.0
| q                                                     |   distinct_answers |
|:------------------------------------------------------|-------------------:|
| farmer asked query on weather                         |              10177 |
| weather forecast                                      |               2328 |
| information regarding weather report                  |               1749 |
| spray schedule in apple                               |               1468 |
| weather report                                        |               1139 |
| regarding spray in apple tree                         |                998 |
| regarding spray schedule of apple                     |                991 |
| regarding spray schedule in apple                     |                818 |
| spray schedule in apple trees                         |                615 |
| todays weather forecast                               |                608 |
| information regarding weather                         |                519 |
| spray schedule in apple orchard                       |                500 |
| spray schedule                                        |                495 |
| spray                                                 |                475 |
| information regarding fertilizer dose of apple plants |                370 |


### Most reused answers

| KccAns                                                                                                                   |   count |
|:-------------------------------------------------------------------------------------------------------------------------|--------:|
| NA                                                                                                                       |    2895 |
| सेंजोस स्केल कीट मार्च माह में सक्रिय हो जाता है। सेब में आधा इंच पत्ती ( हरी कली ) आने से टाइट क्लस्टर अवस्था ( कलियों  |    1331 |
| सेब के 10 या 10 साल से ऊपर के पौधे में 100 किलो गोबर की खाद में सिंगल सुपर फॉस्फेट 2 किलो + 1170 ग्राम म्यूरेट ऑफ़ पोटास  |    1268 |
| spray of pink stage in apple are mancozeb + carbendazim (Combi Product) @ 500gm and biozyme @ 400ml/200 litre of water a |    1071 |
| Bordeaux mixture ratios in apple plants are spray Copper Sulphate @ 1 Kg + Quick Lime @ 1 Kg/100 Liters of water.        |    1040 |
| सेब मे फल विकास (अखरोट के आकार के 20 दिन बाद ) पर असामयिक पतझड़, स्कैब रोग के नियंत्रण हेतु 80 ग्राम नेटिवो(टेबुकोनाजोज+ट |    1018 |
| Dose of spray after harvest in apple is Chlorpyriphos @ 400ml/200 liter of water and Copper Ox chloride @ 600gram/200 li |     998 |
| सेब मे गुलाबी कली अवस्था पर प्रोपिनेब 70WP(एन्ट्राकोल) @ 600 ग्राम या  मैंकोजेब(Mancozeb 75WP) @ 600ग्राम या स्कोर(Difen |     986 |
| Spray With Nativo(Tebuconazole + Trifloxystrobin) @ 80 gm or  Hexaconazole @ 100 ml/200 lt water  & After 1-2 days spray |     787 |
| Triadimefon 25WP (50 g) or                                                                                               |     766 |
| Dodine 65 WP (60 g) or                                                                                                   |         |
| Captan 32% +Tebuconazole 8% 40SC (150 ml) or                                                                             |         |
| Metiram 55% + Pyraclos                                                                                                   |         |
| सेब मे पँखुड़ीपात अवस्था मे टेबुकोनाजोज+ट्राईफ्लोकिसस्ट्रोबीन (नेटिवों) 80 ग्राम,Tebuconazole 8% + Captan 32% SC (Shamir) |     760 |
| Spray dose of Fruit development (walnut size)in apple is  Mancozeb @ 600 gm or Propineb@600gm or Myclobutanil @ 80 gm /2 |     752 |
| Fertilizer dose of apple plants are 100kg FYM + 2800gm CAN (February- march month)+ 2200gm SP + 1200gm MOP (December mon |     750 |
| NUMBER OF NAUNI UNIVERSITY PLEASE CONTACT 01792 252328, 26                                                               |     744 |
| Spray dose of Fruit development (walnut size) stage in apple are Combi-Product (Carbendazim+ Mancozeb) 500gm/200 litre o |     710 |
