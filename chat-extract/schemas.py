from typing import Literal

from pydantic import BaseModel


class DatesResult(BaseModel):
    submission_date: str | None
    ready_date: str | None
    travel_start: str | None
    travel_end: str | None


class VisaTypeResult(BaseModel):
    visa_type: Literal["tourist", "business", "family_visit", "family_reunion", "national_d", "student", "unknown"]


class EntriesResult(BaseModel):
    entry_type: Literal["single", "double", "multi"] | None
    entries_count: int | None


class VisaValidityResult(BaseModel):
    validity_days: int | None
    exactly_under_dates: bool | None


class EmploymentResult(BaseModel):
    employment_status: Literal["self_employed", "employed", "unemployed", "student", "retired", "sponsored", "unknown"]


class GroupResult(BaseModel):
    total_applicants: int
    has_children: bool
    family_group: bool


class SponsorResult(BaseModel):
    has_sponsorship: bool
    sponsor_relation: Literal["spouse", "parents", "relatives", "self_is_sponsor", "unknown_sponsor"] | None


class TripPurposeResult(BaseModel):
    trip_purpose: Literal["tourism", "business", "family_visit", "education", "event", "mixed", "unknown"]
    secondary_purpose: Literal["tourism", "business", "family_visit", "education", "event", "mixed", "unknown"] | None


class VisaCenterResult(BaseModel):
    visa_center: Literal["visametric", "tls", "embassy_direct", "unknown"]


class PriorSchengenResult(BaseModel):
    prior_schengen_count: int | None
    is_first_ever: bool


class RejectionResult(BaseModel):
    rejection_codes: list[int]
    rejection_reasons: list[str]
    rejection_reason_raw: str | None


class CountriesResult(BaseModel):
    primary_country: str
    additional_countries: list[str]
    is_multi_country: bool


class AdditionalDocsResult(BaseModel):
    additional_docs_requested: bool | None
    docs_requested_list: list[str]


class VnjTypeResult(BaseModel):
    vnj_type: Literal["work", "family_reunification", "talent", "property", "student", "self_employment", "unknown"] | None
    vnj_duration_months: int | None
    vnj_under_6_months: bool | None


class RequestedVsGrantedResult(BaseModel):
    mismatch_detected: bool | None
    requested_duration_days: int | None
    granted_duration_days: int | None
    requested_multi: bool | None
    granted_multi: bool | None
