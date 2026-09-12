## Method-comparison data dictionary

The main fields in `wenchuan_method_comparison_results.csv` are:

- record and event descriptors: `RSN`, `Event_Type`, `Origin_Time`, `Station`,
  `Component`, `Record`, `Earthquake_Name`, and `Earthquake_Magnitude`;
- site and distance metadata: `EpiD_km`, `HypD_km`,
  `Joyner_Boore_Distance_km`, `Rx`, `Vs30_m_per_s`,
  `Preferred_NEHRP_Class`, and `Instrument_Model`;
- processing descriptors: `HPCF_PEER`, `LPCF_PEER`, `nroll`,
  `P_arrival_time_s`, `PGA`, and `MS`;
- method outputs: `hpcf_snr`, `hpcf_ridder`, `hpcf_RS`, `hpcf_Guo`, and
  `hpcf_EMr`.

`Event_Type` is represented in English as `Mainshock` or `Aftershock`. Column
names were translated to English without altering the stored numerical values.

