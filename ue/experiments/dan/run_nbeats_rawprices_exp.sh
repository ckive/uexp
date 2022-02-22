contracts=("BTCUSDT" "ETHUSDT" "DOGEUSDT")
max_encoder_lengths=(30 60 120 180 300)

for contract in ${contracts[@]}; do
  for mel in ${max_encoder_lengths[@]};do
    nohup python run_nbeats_rawprices.py $contract $mel &
  done
done