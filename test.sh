
rm offline_imgs/fill_q_handheld_male_rect_test/*.png
rm offline_q_test/*
rm capture/fill_q_handheld_male_rect_test.csv
# python scripts/run_offline_pdr.py --csv capture/handheld_male_rect.csv --calib calib.csv --output-dir offline_q_test --zupt --lp
# cp offline_q_test/fill_q_handheld_male_rect.csv capture/fill_q_handheld_male_rect_test.csv
# python debug/run_zupt.py 
# open offline_imgs/fill_q_handheld_male_rect_test/ # run_zupt.py

open offline_q_test/ # run_offlien_pdr 
python scripts/run_offline_pdr_for_util.py --csv capture/full_handheld_nifei_rect.csv --calib calib.csv --output-dir offline_q_test --zupt --lp
# python scripts/run_offline_pdr.py --csv capture/handheld_nifei_init.csv --calib calib.csv --output-dir offline_q_test --zupt --lp
# cp offline_q_test/fill_q_handheld_nifei_init.csv capture/fill_q_handheld_nifei_init.csv
# python debug/run_zupt.py 
