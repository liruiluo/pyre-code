import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';

export const dynamic = 'force-dynamic';

const COOKIE_AGE = 60 * 60 * 24 * 365;

async function getSession() {
  const cookieStore = await cookies();
  const existing = cookieStore.get('session_token')?.value;
  const token = existing || crypto.randomUUID();

  // A learner should be able to organize a problem before their first submission.
  const userResponse = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken: token }),
    cache: 'no-store',
  });
  if (!userResponse.ok) throw new Error('Unable to initialize learner session');
  const { userId } = (await userResponse.json()) as { userId: number };
  return { userId, token };
}

async function relay(response: Response, token: string) {
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    data = { detail: 'Study list service returned an invalid response' };
  }
  const result = NextResponse.json(data, { status: response.status });
  result.headers.set('Cache-Control', 'no-store');
  // Renew the anonymous session on every visit so study lists remain reachable.
  result.cookies.set('session_token', token, {
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      maxAge: COOKIE_AGE,
      path: '/',
  });
  return result;
}

export async function GET() {
  try {
    const { userId, token } = await getSession();
    const upstream = await fetch(`${GRADING_SERVICE_URL}/study-lists/${userId}`, {
      cache: 'no-store',
    });
    return relay(upstream, token);
  } catch {
    return NextResponse.json({ detail: 'Study list service unavailable' }, { status: 502 });
  }
}

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      throw new Error('Invalid body');
    }
  } catch {
    return NextResponse.json({ detail: 'Invalid JSON body' }, { status: 400 });
  }

  try {
    const { token } = await getSession();
    const upstream = await fetch(`${GRADING_SERVICE_URL}/study-lists/actions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, sessionToken: token }),
      cache: 'no-store',
    });
    return relay(upstream, token);
  } catch {
    return NextResponse.json({ detail: 'Study list service unavailable' }, { status: 502 });
  }
}
