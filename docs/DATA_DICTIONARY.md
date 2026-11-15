# Data dictionary (core fields)

All timestamps are ISO-8601 UTC strings in CSV exports. IDs are immutable strings.

| Entity | Primary key | Important fields | Validation |
|---|---|---|---|
| Carrier | carrier_id | carrier_name, on_time_rate, active | reliability 0..1 |
| Facility | facility_id | city, capacity_per_hour, timezone | capacity > 0 |
| Route | route_id | origin_facility_id, destination_facility_id, distance_km, planned_hours | both facility FKs; origin != destination |
| ServiceLevel | service_level_id | allowed_delay_hours, target_hours | hours >= 0 |
| Shipment | shipment_id | carrier_id, route_id, service_level_id, created_at, planned_delivery_at, actual_delivery_at, current_status, is_delayed | valid FKs; delivery >= created; status enum |
| ShipmentLeg | leg_id | shipment_id, route_id, origin_facility_id, destination_facility_id, leg_sequence, leg_status | valid FKs; sequence > 0 |
| TrackingEvent | event_id | shipment_id, event_type, event_time, ingestion_time, facility_id, event_sequence | valid FKs; ingestion >= event time; sequence > 0 |
| FacilityQueue | queue_id | facility_id, observed_at, queue_length, avg_service_minutes, capacity, queue_pressure | nonnegative counts; pressure = queue_length/capacity |
| WeatherEvent | weather_event_id | facility_id, start_at, end_at, severity, rainfall_mm | end > start; severity LOW/MODERATE/HIGH/SEVERE |
| RouteWeather | route_weather_id | route_id, weather_event_id | bridge table; both FKs |
| ETAEstimate | eta_id | shipment_id, prediction_time, input features, predicted_remaining_hours, actual_remaining_hours, delay_label | snapshot cannot use events after prediction_time |
| DelayReason | delay_reason_id | shipment_id, reason_code, reason_description | reason code from documented set |

## Cardinality
Carrier 1:N Shipment; Route 1:N ShipmentLeg; Shipment 1:N ShipmentLeg;
Shipment 1:N TrackingEvent; Facility 1:N FacilityQueue; Shipment 1:N ETAEstimate;
Route N:M WeatherEvent through RouteWeather; Shipment 0:1 DelayReason in this generated sample.