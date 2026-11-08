from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    id: str
    state: str
    city: str
    latitude: float
    longitude: float
    dataset_state: str


_LOCATIONS_LIST = [
    Location("johor", "Johor", "Johor Bahru", 1.4927, 103.7414, "Johor (Johor Bahru)"),
    Location("kedah", "Kedah", "Alor Setar", 6.1248, 100.3678, "Kedah (Alor Setar)"),
    Location("kelantan", "Kelantan", "Kota Bharu", 6.1254, 102.2381, "Kelantan (Kota Bharu)"),
    Location("kuala_lumpur", "Kuala Lumpur", "Kuala Lumpur", 3.1319, 101.6841, "Kuala Lumpur"),
    Location("melaka", "Melaka", "Bandar Melaka", 2.1896, 102.2501, "Melaka (Bandar Melaka)"),
    Location("negeri_sembilan", "Negeri Sembilan", "Seremban", 2.7258, 101.9424, "Negeri Sembilan (Seremban)"),
    Location("pahang", "Pahang", "Kuantan", 3.8126, 103.3256, "Pahang (Kuantan)"),
    Location("penang", "Penang", "George Town", 5.4141, 100.3288, "Penang (George Town)"),
    Location("perak", "Perak", "Ipoh", 4.5975, 101.0901, "Perak (Ipoh)"),
    Location("perlis", "Perlis", "Kangar", 6.4412, 100.1986, "Perlis (Kangar)"),
    Location("putrajaya", "Putrajaya", "Putrajaya", 2.9264, 101.6964, "Putrajaya"),
    Location("sabah", "Sabah", "Kota Kinabalu", 5.9804, 116.0735, "Sabah (Kota Kinabalu)"),
    Location("sarawak", "Sarawak", "Kuching", 1.5533, 110.3592, "Sarawak (Kuching)"),
    Location("selangor", "Selangor", "Shah Alam", 3.0738, 101.5183, "Selangor (Shah Alam)"),
    Location("terengganu", "Terengganu", "Kuala Terengganu", 5.3117, 103.1324, "Terengganu (K. Terengganu)"),
]

LOCATIONS: dict[str, Location] = {loc.id: loc for loc in _LOCATIONS_LIST}
