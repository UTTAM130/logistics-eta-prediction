PRAGMA foreign_keys = ON;
CREATE TABLE Carrier(carrier_id TEXT PRIMARY KEY, carrier_name TEXT NOT NULL,
 on_time_rate REAL NOT NULL CHECK(on_time_rate BETWEEN 0 AND 1), active INTEGER NOT NULL);
CREATE TABLE Facility(facility_id TEXT PRIMARY KEY, facility_name TEXT NOT NULL,
 city TEXT NOT NULL, capacity_per_hour INTEGER NOT NULL CHECK(capacity_per_hour > 0));
CREATE TABLE Route(route_id TEXT PRIMARY KEY, origin_facility_id TEXT NOT NULL,
 destination_facility_id TEXT NOT NULL, distance_km REAL NOT NULL CHECK(distance_km > 0),
 planned_hours REAL NOT NULL CHECK(planned_hours > 0),
 FOREIGN KEY(origin_facility_id) REFERENCES Facility(facility_id),
 FOREIGN KEY(destination_facility_id) REFERENCES Facility(facility_id));
CREATE TABLE ServiceLevel(service_level_id TEXT PRIMARY KEY, service_name TEXT NOT NULL,
 allowed_delay_hours REAL NOT NULL, target_hours REAL NOT NULL);
CREATE TABLE Shipment(shipment_id TEXT PRIMARY KEY, carrier_id TEXT NOT NULL, route_id TEXT NOT NULL,
 service_level_id TEXT NOT NULL, created_at TEXT NOT NULL, planned_delivery_at TEXT NOT NULL,
 actual_delivery_at TEXT NOT NULL, current_status TEXT NOT NULL,
 FOREIGN KEY(carrier_id) REFERENCES Carrier(carrier_id),
 FOREIGN KEY(route_id) REFERENCES Route(route_id),
 FOREIGN KEY(service_level_id) REFERENCES ServiceLevel(service_level_id));
CREATE TABLE ShipmentLeg(leg_id TEXT PRIMARY KEY, shipment_id TEXT NOT NULL, route_id TEXT NOT NULL,
 leg_sequence INTEGER NOT NULL, leg_status TEXT NOT NULL,
 FOREIGN KEY(shipment_id) REFERENCES Shipment(shipment_id),
 FOREIGN KEY(route_id) REFERENCES Route(route_id));
CREATE TABLE TrackingEvent(event_id TEXT PRIMARY KEY, shipment_id TEXT NOT NULL,
 event_type TEXT NOT NULL, event_time TEXT NOT NULL, ingestion_time TEXT NOT NULL,
 facility_id TEXT NOT NULL, event_sequence INTEGER NOT NULL,
 FOREIGN KEY(shipment_id) REFERENCES Shipment(shipment_id),
 FOREIGN KEY(facility_id) REFERENCES Facility(facility_id));
CREATE TABLE FacilityQueue(queue_id TEXT PRIMARY KEY, facility_id TEXT NOT NULL,
 observed_at TEXT NOT NULL, queue_length INTEGER NOT NULL, capacity INTEGER NOT NULL,
 FOREIGN KEY(facility_id) REFERENCES Facility(facility_id));
CREATE TABLE WeatherEvent(weather_event_id TEXT PRIMARY KEY, facility_id TEXT NOT NULL,
 start_at TEXT NOT NULL, end_at TEXT NOT NULL, severity TEXT NOT NULL);
CREATE TABLE RouteWeather(route_weather_id TEXT PRIMARY KEY, route_id TEXT NOT NULL,
 weather_event_id TEXT NOT NULL, FOREIGN KEY(route_id) REFERENCES Route(route_id),
 FOREIGN KEY(weather_event_id) REFERENCES WeatherEvent(weather_event_id));