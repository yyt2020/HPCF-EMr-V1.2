# User-supplied input files

Raw China Earthquake Networks Center (CENC) strong-motion records are not
included in this repository because the authors do not hold redistribution
rights. Researchers may request access through the National Earthquake Science
Data Center at <https://data.earthquake.cn/>, subject to the Center's applicable
data-access conditions.

After obtaining authorized copies, create this layout:

```text
input/
|-- station_metadata.csv
`-- acceleration_records/
    |-- record_001.dat
    `-- ...
```

Copy `station_metadata_template.csv` to `station_metadata.csv` and add one row
per record. Do not change the column order: the current implementation reads
the physical filename from column 5 (`Record`) and the event label from column
7 (`Earthquake_Name`). Each `Record` value must exactly match a `.dat` filename
inside `acceleration_records/`.

Metadata fields are:

| Column | Meaning |
|---|---|
| `Event_Type` | Mainshock/aftershock or other event category |
| `Origin_Time` | Earthquake origin time |
| `Station` | Station identifier |
| `Component` | Motion component |
| `Record` | Exact physical filename, including `.dat` |
| `Minimum_Usable_Frequency_Hz` | Reference lower usable frequency |
| `Earthquake_Name` | Event name used for grouping |
| `Earthquake_Magnitude` | Event magnitude |
| `nroll` | Roll-off parameter retained from study metadata |
| `Epicentral_Distance_km` | Epicentral distance |
| `Hypocentral_Distance_km` | Hypocentral distance |
| `Joyner_Boore_Distance_km` | Joyner-Boore distance |
| `Rx_km` | Rx distance |
| `Vs30_m_per_s` | Time-averaged shear-wave velocity in the upper 30 m |
| `Preferred_NEHRP_Class` | NEHRP site class |
| `Instrument_Model` | Recording instrument model |
| `MS` | Study-specific metadata field retained for traceability |

The CENC text reader used by this code parses the sampling interval from line
12 and acceleration samples beginning on line 17. Records in another format
must be converted by an authorized user without distributing restricted source
data.
