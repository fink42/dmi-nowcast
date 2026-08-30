/** Outbound links, in one place so they cannot drift between pages. */
export const GITHUB_URL = 'https://github.com/nsimonsen/dmi-nowcast';
export const GITHUB_ISSUES_URL = `${GITHUB_URL}/issues`;
export const DMI_OPEN_DATA_URL = 'https://opendatadocs.dmi.govcloud.dk/';
export const DMI_URL = 'https://www.dmi.dk/';
export const OSM_COPYRIGHT_URL = 'https://www.openstreetmap.org/copyright';
export const PROTOMAPS_URL = 'https://protomaps.com/';
export const NATURAL_EARTH_URL = 'https://www.naturalearthdata.com/';

/**
 * Donation links — deliberately inactive for launch (Phase C decision: the
 * support page ships with the slots commented out and a note). Uncomment and
 * fill in when the accounts exist; the support page renders whatever is here.
 */
export const DONATION_LINKS: { label: string; url: string }[] = [
	// { label: 'GitHub Sponsors', url: 'https://github.com/sponsors/nsimonsen' },
	// { label: 'Ko-fi', url: 'https://ko-fi.com/…' }
];
