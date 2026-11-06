# Sample design

- split: test, eligible: 8,956, sampled: 1,000
- seed: 42; candidates capped at 30 per query (all E kept)
- truncated queries: 148

## Department

department
Sports & Outdoors             90
Health & Household            85
Baby & Other                  85
Grocery                       85
Tools, Automotive & Garden    83
Office, Books & Media         83
Beauty & Personal Care        83
Home & Kitchen                83
Clothing, Shoes & Jewelry     83
Electronics                   81
Toys & Games                  80
Pet Supplies                  79

## Brand query

is_brand_query
False    812
True     188

## Difficulty (E share of candidates)

difficulty
medium    587
hard      304
easy      109

## Eligible-population marginals (for comparison)

department
Sports & Outdoors             0.090
Baby & Other                  0.085
Grocery                       0.085
Health & Household            0.085
Clothing, Shoes & Jewelry     0.083
Beauty & Personal Care        0.083
Home & Kitchen                0.083
Tools, Automotive & Garden    0.083
Office, Books & Media         0.083
Electronics                   0.081
Toys & Games                  0.080
Pet Supplies                  0.079

difficulty
medium    0.574
hard      0.318
easy      0.108