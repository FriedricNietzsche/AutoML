import { useEffect, useState } from 'react';
import AppShell from '../components/shell/AppShell';
import { getCurrentSession, type BuildSession } from '../lib/buildSession';
import { Redirect } from '../router/router';
import MatrixScreenLoader from '../components/MatrixScreenLoader';

export default function WorkspacePage() {
  const [hydrated, setHydrated] = useState(false);
  const [session, setSession] = useState<BuildSession | null | undefined>(undefined);

  useEffect(() => {
    setHydrated(true);
    setSession(getCurrentSession());
  }, []);

  if (!hydrated || session === undefined) {
    return <MatrixScreenLoader label="Loading workspace…" />;
  }

  // Guard: require a session.
  if (!session) {
    return <Redirect to="/" />;
  }

  return <AppShell session={session} />;
}
